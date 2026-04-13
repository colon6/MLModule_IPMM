import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client
from dotenv import load_dotenv
import os
import uuid
import time
import traceback

load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

supabase = create_client(url, key)

model = SentenceTransformer("all-MiniLM-L6-v2")

TOP_PROJECTS_NUM = 3

print("Worker started. Waiting for jobs...")


def run_matching(chosen_term, job_id):

    archive_resp = supabase.table("results_tab") \
        .update({"status": "archived"}) \
        .eq("term", chosen_term) \
        .execute()
    print("Archived old results:", archive_resp)


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

    print("Matching completed for:", chosen_term)
    return True

if __name__ == "__main__":

    while True:

        try:

            jobs = (
                supabase.table("jobs")
                .select("*")
                .eq("status", "ready")
                .limit(1)
                .execute()
            )

            if not jobs.data:
                print("No jobs found...")
                time.sleep(10)
                continue

            job = jobs.data[0]

            job_id = job["id"]
            chosen_term = job["term"]

            print("Running job:", job_id)

            # mark job running
            supabase.table("jobs").update({
                "status": "processing",
                "python_error": None
            }).eq("id", job_id).execute()

            try:

                run_matching(chosen_term, job_id)

                # mark completed
                supabase.table("jobs").update({
                    "status": "completed"
                }).eq("id", job_id).execute()

                print("Job finished:", job_id)

            except Exception as job_error:

                error_text = traceback.format_exc()

                print("Job failed:", job_id)
                print(error_text)

                supabase.table("jobs").update({
                    "status": "failed",
                    "python_error": error_text
                }).eq("id", job_id).execute()

        except Exception as worker_error:

            print("Worker error:", worker_error)

        time.sleep(10)