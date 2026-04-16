import os
import uuid
import traceback
import numpy as np
from fastapi import FastAPI, BackgroundTasks, Request, HTTPException
import onnxruntime as ort
from tokenizers import Tokenizer
from supabase import create_client
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()

app = FastAPI(title="AI Matching Worker")

# 2. Global Resource Initialization
URL = os.environ.get("SUPABASE_URL")
KEY = os.environ.get("SUPABASE_KEY")

if not URL or not KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY environment variables")

supabase = create_client(URL, KEY)

# Load model once at startup — quantized INT8 (~22MB vs 86MB full)
tokenizer = Tokenizer.from_file("./model/tokenizer.json")
tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
tokenizer.enable_truncation(max_length=512)
session = ort.InferenceSession("./model/onnx/model_qint8.onnx")

TOP_PROJECTS_NUM = 3


def embed(texts: list[str]) -> np.ndarray:
    """Tokenize + ONNX inference with mean pooling and L2 normalization."""
    encoded = tokenizer.encode_batch(texts)
    input_ids      = np.array([e.ids            for e in encoded], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

    token_embeddings = session.run(None, {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    })[0]  # shape: (batch, seq_len, hidden)

    mask       = attention_mask[:, :, np.newaxis].astype(np.float32)
    summed     = np.sum(token_embeddings * mask, axis=1)
    counts     = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    embeddings = summed / counts

    # L2 normalize — cosine similarity becomes a plain dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-9)


# 3. Matching Logic
def run_matching_logic(chosen_term: str, job_id: str):
    """Full matching pipeline — runs in background so webhook returns fast."""
    try:
        print(f"Starting matching for Job ID: {job_id}, Term: {chosen_term}")

        supabase.table("jobs").update({
            "status": "processing",
            "python_error": None
        }).eq("id", job_id).execute()

        # Archive old results for this term
        supabase.table("results_tab") \
            .update({"status": "archived"}) \
            .eq("term", chosen_term) \
            .execute()

        # Fetch data
        interns  = supabase.table("resumes").select("*").eq("term", chosen_term).execute().data
        projects = supabase.table("projects").select("*").eq("term", chosen_term).execute().data

        if not interns or not projects:
            raise Exception(f"Insufficient data: {len(interns)} interns, {len(projects)} projects found.")

        # Build project combined text
        for p in projects:
            p["combined_text"] = " ".join(filter(None, [
                p.get("description", ""),
                p.get("deliverable", ""),
                p.get("requirements", ""),
            ]))

        intern_embeddings  = embed([r["text"]           for r in interns])   # (n_interns, hidden)
        project_embeddings = embed([p["combined_text"]  for p in projects])  # (n_projects, hidden)

        # Cosine similarity = dot product (embeddings are L2 normalized)
        sim_matrix = intern_embeddings @ project_embeddings.T  # (n_interns, n_projects)

        # GALE-SHAPLEY implementation
        n_interns, n_projects = sim_matrix.shape
        
        # Extract project capacities (default to 1 if not exactly specified)
        project_caps = [p.get("intern_cap", 1) for p in projects]
        # Handle case where intern_cap exists in the dictionary but its value is None
        project_caps = [cap if cap is not None else 1 for cap in project_caps]
        
        # Build preference lists for interns (indices of projects sorted by similarity descending)
        intern_prefs = []
        for i in range(n_interns):
            # argsort sorts ascending, so we reverse it
            prefs = np.argsort(sim_matrix[i])[::-1]
            intern_prefs.append(list(prefs))
            
        intern_free = list(range(n_interns))
        intern_next_proposal = [0] * n_interns
        project_matches = {j: [] for j in range(n_projects)} # proj_idx -> list of intern_idx
        
        while intern_free:
            i = intern_free.pop(0)
            
            if intern_next_proposal[i] >= n_projects:
                continue # Exhausted all projects (should only happen if all projects are full)
                
            j = intern_prefs[i][intern_next_proposal[i]]
            intern_next_proposal[i] += 1
            
            score_i = sim_matrix[i, j]
            cap = project_caps[j]
            
            if len(project_matches[j]) < cap:
                project_matches[j].append(i)
                # Sort accepted interns by their score so the worst is at the end
                project_matches[j].sort(key=lambda idx: sim_matrix[idx, j], reverse=True)
            else:
                # Compare with the worst accepted intern (last in the sorted list)
                worst_match_i = project_matches[j][-1]
                if score_i > sim_matrix[worst_match_i, j]:
                    project_matches[j][-1] = i
                    project_matches[j].sort(key=lambda idx: sim_matrix[idx, j], reverse=True)
                    intern_free.append(worst_match_i)
                else:
                    intern_free.append(i)
                    
        # Reverse mapping from intern -> project
        intern_to_project = {}
        for j, matched_interns in project_matches.items():
            for i in matched_interns:
                intern_to_project[i] = j

        match_id = str(uuid.uuid4())
        results  = []

        for i, intern in enumerate(interns):
            sims        = sim_matrix[i]
            top_indices = sims.argsort()[::-1][:TOP_PROJECTS_NUM]

            project_list = [
                {
                    "rank":       rank,
                    "project_id": projects[pi]["id"],
                    "score":      float(sims[pi]),
                }
                for rank, pi in enumerate(top_indices, start=1)
            ]

            assigned_proj_idx = intern_to_project.get(i)
            recommended_id = projects[assigned_proj_idx]["id"] if assigned_proj_idx is not None else None

            results.append({
                "match_id":               match_id,
                "intern_id":              intern["id"],
                "term":                   chosen_term,
                "projects":               project_list,
                "recommended_project_id": recommended_id,
                "status":                 "pending",
            })

        if results:
            supabase.table("results_tab").insert(results).execute()

        supabase.table("jobs").update({"status": "completed"}).eq("id", job_id).execute()
        print(f"Successfully completed Job: {job_id}")

    except Exception:
        error_text = traceback.format_exc()
        print(f"Error in Job {job_id}: {error_text}")
        supabase.table("jobs").update({
            "status": "failed",
            "python_error": error_text
        }).eq("id", job_id).execute()


# 4. Endpoints
@app.post("/webhook")
async def handle_supabase_trigger(request: Request, background_tasks: BackgroundTasks):
    """
    Receives payload from Supabase.
    Expects JSON: {"record": {"id": "...", "status": "ready", "term": "..."}}
    """
    try:
        payload = await request.json()
        record  = payload.get("record", {})

        job_id = record.get("id")
        status = record.get("status")
        term   = record.get("term")

        if status == "ready" and job_id and term:
            background_tasks.add_task(run_matching_logic, term, job_id)
            return {"message": "Job received and background processing started", "job_id": job_id}

        return {"message": "Webhook received but ignored (status not 'ready' or missing data)"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "online", "model": "all-MiniLM-L6-v2-qint8"}
