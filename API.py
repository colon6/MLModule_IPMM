from fastapi import FastAPI, Request
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client
from dotenv import load_dotenv
import os
import uuid
import traceback

load_dotenv()

app = FastAPI()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

supabase = create_client(url, key)

model = SentenceTransformer("all-MiniLM-L6-v2")

TOP_PROJECTS_NUM = 3


def run_matching(chosen_term, job_id):

    supabase.table("results_tab") \
        .update({"status": "archived"}) \
        .eq("term", chosen_term) \
        .execute()

    match_id = str(uuid.uuid4())

    interns_by_term = supabase.table("resumes").select("*").eq("term", chosen_term).execute()
    projects_by_term = supabase.table("projects").select("*").eq("term", chosen_term).execute()

    interns_df = pd.DataFrame(interns_by_term.data)
    projects_df = pd.DataFrame(projects_by_term.data)

    if interns_df.empty or projects_df.empty:
        raise Exception("No interns or projects found for this term")

    projects_df["combined_text"] = (
        projects_df[["description", "deliverable", "requirements"]]
        .fillna("")
        .agg(" ".join, axis=1)
    )

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

    return True


# 🚀 THIS replaces your while loop
@app.post("/run-job")
async def run_job(request: Request):

    try:
        data = await request.json()

        job_id = data.get("job_id")
        chosen_term = data.get("term")

        print("Running job:", job_id)

        supabase.table("jobs").update({
            "status": "processing",
            "python_error": None
        }).eq("id", job_id).execute()

        try:
            run_matching(chosen_term, job_id)

            supabase.table("jobs").update({
                "status": "completed"
            }).eq("id", job_id).execute()

            return {"status": "completed"}

        except Exception:
            error_text = traceback.format_exc()

            supabase.table("jobs").update({
                "status": "failed",
                "python_error": error_text
            }).eq("id", job_id).execute()

            return {"status": "failed", "error": error_text}

    except Exception as e:
        return {"error": str(e)}