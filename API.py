from fastapi import FastAPI, Request, BackgroundTasks
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client
from dotenv import load_dotenv
import os
import uuid
import traceback

# =========================
# LOAD ENV
# =========================
load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

if not url or not key:
    raise Exception("Missing SUPABASE_URL or SUPABASE_KEY")

supabase = create_client(url, key)

print("🚀 Supabase connected")

# =========================
# FASTAPI APP
# =========================
app = FastAPI()

print("🚀 API starting...")

# =========================
# LAZY MODEL LOADING
# =========================
model = None

def get_model():
    global model
    if model is None:
        print("📦 Loading SentenceTransformer model...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Model loaded")
    return model


TOP_PROJECTS_NUM = 3


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def root():
    print("Health check hit")
    return {"status": "API is running"}


# =========================
# CORE MATCHING LOGIC
# =========================
def run_matching(chosen_term, job_id):

    print(f"⚙️ Starting matching for job {job_id}")

    # Archive old results
    supabase.table("results_tab") \
        .update({"status": "archived"}) \
        .eq("term", chosen_term) \
        .execute()

    match_id = str(uuid.uuid4())

    interns_by_term = (
        supabase.table("resumes")
        .select("*")
        .eq("term", chosen_term)
        .execute()
    )

    projects_by_term = (
        supabase.table("projects")
        .select("*")
        .eq("term", chosen_term)
        .execute()
    )

    interns_df = pd.DataFrame(interns_by_term.data or [])
    projects_df = pd.DataFrame(projects_by_term.data or [])

    if interns_df.empty or projects_df.empty:
        raise Exception("No interns or projects found for this term")

    # Combine project text
    projects_df["combined_text"] = (
        projects_df[["description", "deliverable", "requirements"]]
        .fillna("")
        .agg(" ".join, axis=1)
    )

    # Load model safely
    model = get_model()

    intern_embeddings = model.encode(
        interns_df["text"].tolist(),
        convert_to_numpy=True
    )

    project_embeddings = model.encode(
        projects_df["combined_text"].tolist(),
        convert_to_numpy=True
    )

    results = []

    for intern_idx, intern_row in interns_df.iterrows():

        similarities = cosine_similarity(
            [intern_embeddings[intern_idx]],
            project_embeddings
        )[0]

        top_indices = similarities.argsort()[::-1][:TOP_PROJECTS_NUM]
        top_scores = similarities[top_indices]

        project_list = []

        for rank, project_idx in enumerate(top_indices, start=1):

            project_row = projects_df.iloc[project_idx]

            project_list.append({
                "rank": rank,
                "project_id": project_row["id"],
                "score": float(top_scores[rank - 1]),
                "text": project_row["combined_text"]
            })

        results.append({
            "match_id": match_id,
            "intern_id": intern_row["id"],
            "term": chosen_term,
            "projects": project_list,
            "recommended_project_id": project_list[0]["project_id"],
            "status": "pending"
        })

    if results:
        supabase.table("results_tab").insert(results).execute()

    print(f"✅ Matching completed for {job_id}")


# =========================
# WEBHOOK ENDPOINT
# =========================
@app.post("/run-job")
async def run_job(request: Request, background_tasks: BackgroundTasks):

    payload = await request.json()
    record = payload.get("record", {})

    job_id = record.get("id")
    chosen_term = record.get("term")
    status = record.get("status")

    print("Webhook received:", record)

    # Ignore non-ready jobs
    if status != "ready":
        return {"message": "ignored (not ready)"}

    # Mark processing
    supabase.table("jobs").update({
        "status": "processing",
        "python_error": None
    }).eq("id", job_id).execute()

    # Background processing
    def process_job():
        try:
            run_matching(chosen_term, job_id)

            supabase.table("jobs").update({
                "status": "completed"
            }).eq("id", job_id).execute()

            print("✅ Job completed:", job_id)

        except Exception:
            error_text = traceback.format_exc()

            print("❌ Job failed:", job_id)
            print(error_text)

            supabase.table("jobs").update({
                "status": "failed",
                "python_error": error_text
            }).eq("id", job_id).execute()

    background_tasks.add_task(process_job)

    return {"message": "job accepted"}