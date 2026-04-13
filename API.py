import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client
from dotenv import load_dotenv
import os
import uuid
import traceback

from fastapi import FastAPI, BackgroundTasks, Request

# -----------------------
# Setup
# -----------------------
load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

supabase = create_client(url, key)

model = SentenceTransformer("all-MiniLM-L6-v2")

TOP_PROJECTS_NUM = 3

app = FastAPI()

print("API started - waiting for webhook calls...")


# -----------------------
# Core Matching Logic
# -----------------------
def run_matching(chosen_term, job_id):

    # archive old results
    supabase.table("results_tab") \
        .update({"status": "archived"}) \
        .eq("term", chosen_term) \
        .execute()

    match_id = str(uuid.uuid4())

    # fetch data
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

    interns_df = pd.DataFrame(interns_by_term.data)
    projects_df = pd.DataFrame(projects_by_term.data)

    if interns_df.empty or projects_df.empty:
        raise Exception("No interns or projects found for this term")

    # combine project text
    projects_df["combined_text"] = (
        projects_df[["description", "deliverable", "requirements"]]
        .fillna("")
        .agg(" ".join, axis=1)
    )

    # embeddings
    intern_embeddings = model.encode(
        interns_df["text"].tolist(),
        convert_to_numpy=True
    )

    project_embeddings = model.encode(
        projects_df["combined_text"].tolist(),
        convert_to_numpy=True
    )

    results = []

    # matching loop
    for intern_idx, intern_row in interns_df.iterrows():

        intern_vector = intern_embeddings[intern_idx]

        similarities = cosine_similarity(
            [intern_vector],
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

    print(f"Matching completed for term: {chosen_term}")


# -----------------------
# API Endpoint (Webhook)
# -----------------------
@app.post("/run-job")
async def run_job(request: Request, background_tasks: BackgroundTasks):

    try:
        payload = await request.json()

        record = payload.get("record", {})

        job_id = record.get("id")
        chosen_term = record.get("term")
        status = record.get("status")

        # only process "ready" jobs
        if status != "ready":
            return {"message": "ignored (status not ready)"}

        # mark job as processing
        supabase.table("jobs").update({
            "status": "processing",
            "python_error": None
        }).eq("id", job_id).execute()

        # run in background (non-blocking response)
        background_tasks.add_task(process_job, job_id, chosen_term)

        return {
            "message": "job accepted",
            "job_id": job_id
        }

    except Exception:
        error_text = traceback.format_exc()

        print("Webhook error:", error_text)

        return {
            "status": "error",
            "error": error_text
        }


# -----------------------
# Background Worker
# -----------------------
def process_job(job_id, chosen_term):

    try:
        run_matching(chosen_term, job_id)

        supabase.table("jobs").update({
            "status": "completed"
        }).eq("id", job_id).execute()

        print("Job completed:", job_id)

    except Exception:
        error_text = traceback.format_exc()

        supabase.table("jobs").update({
            "status": "failed",
            "python_error": error_text
        }).eq("id", job_id).execute()

        print("Job failed:", error_text)