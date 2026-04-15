import os
import uuid
import traceback
import pandas as pd
from fastapi import FastAPI, BackgroundTasks, Request, HTTPException
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()

app = FastAPI(title="AI Matching Worker")

# 2. Global Resource Initialization
# Loading these outside the endpoint ensures they stay in memory
URL = os.environ.get("SUPABASE_URL")
KEY = os.environ.get("SUPABASE_KEY")

if not URL or not KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY environment variables")

supabase = create_client(URL, KEY)

# Load the AI model into memory once when the server starts
print("Loading AI Model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded and ready.")

TOP_PROJECTS_NUM = 3


# 3. The Heavy Lifting (Background Logic)
def run_matching_logic(chosen_term: str, job_id: str):
    """Isolated logic to process AI matching without blocking the API"""
    try:
        print(f"Starting matching for Job ID: {job_id}, Term: {chosen_term}")

        # Mark job as processing
        supabase.table("jobs").update({
            "status": "processing",
            "python_error": None
        }).eq("id", job_id).execute()

        # Archive old results for this term
        supabase.table("results_tab") \
            .update({"status": "archived"}) \
            .eq("term", chosen_term) \
            .execute()

        # Fetch Data
        interns_by_term = supabase.table("resumes").select("*").eq("term", chosen_term).execute()
        projects_by_term = supabase.table("projects").select("*").eq("term", chosen_term).execute()

        interns_df = pd.DataFrame(interns_by_term.data)
        projects_df = pd.DataFrame(projects_by_term.data)

        if interns_df.empty or projects_df.empty:
            raise Exception(f"Insufficient data: {len(interns_df)} interns, {len(projects_df)} projects found.")

        # AI Processing
        projects_df["combined_text"] = (
            projects_df[["description", "deliverable", "requirements"]]
            .fillna("")
            .agg(" ".join, axis=1)
        )

        intern_embeddings = model.encode(interns_df["text"].tolist(), convert_to_numpy=True)
        project_embeddings = model.encode(projects_df["combined_text"].tolist(), convert_to_numpy=True)

        match_id = str(uuid.uuid4())
        results = []

        for intern_idx, intern_row in interns_df.iterrows():
            intern_vector = intern_embeddings[intern_idx]
            similarities = cosine_similarity([intern_vector], project_embeddings)[0]

            top_indices = similarities.argsort()[::-1][:TOP_PROJECTS_NUM]
            top_scores = similarities[top_indices]

            project_list = []
            for rank, project_idx in enumerate(top_indices, start=1):
                project_row = projects_df.iloc[project_idx]
                project_list.append({
                    "rank": rank,
                    "project_id": project_row["id"],
                    "score": float(top_scores[rank - 1])
                })

            results.append({
                "match_id": match_id,
                "intern_id": intern_row["id"],
                "term": chosen_term,
                "projects": project_list,
                "recommended_project_id": project_list[0]["project_id"],
                "status": "pending"
            })

        # Insert results and finalize job
        if results:
            supabase.table("results_tab").insert(results).execute()

        supabase.table("jobs").update({"status": "completed"}).eq("id", job_id).execute()
        print(f"Successfully completed Job: {job_id}")

    except Exception as e:
        error_text = traceback.format_exc()
        print(f"Error in Job {job_id}: {error_text}")
        supabase.table("jobs").update({
            "status": "failed",
            "python_error": error_text
        }).eq("id", job_id).execute()


# 4. The Webhook Endpoint
@app.post("/webhook")
async def handle_supabase_trigger(request: Request, background_tasks: BackgroundTasks):
    """
    Receives payload from Supabase.
    Expects JSON format: {"record": {"id": "...", "status": "ready", "term": "..."}}
    """
    try:
        payload = await request.json()
        record = payload.get("record", {})

        job_id = record.get("id")
        status = record.get("status")
        term = record.get("term")

        # Security/Logic Check: Only run if status is 'ready'
        if status == "ready" and job_id and term:
            # Add to background tasks so API returns 200 immediately
            background_tasks.add_task(run_matching_logic, term, job_id)
            return {"message": "Job received and background processing started", "job_id": job_id}

        return {"message": "Webhook received but ignored (status not 'ready' or missing data)"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "online", "model": "all-MiniLM-L6-v2"}
