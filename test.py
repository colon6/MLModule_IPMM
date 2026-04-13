import pytest
import numpy as np

# 👉 CHANGE THIS to your actual file name
from draft_2 import run_matching


# -----------------------
# Mock Supabase Classes
# -----------------------

class MockResponse:
    def __init__(self, data):
        self.data = data


class MockTable:
    def __init__(self, name, db):
        self.name = name
        self.db = db
        self.query = {}

    def select(self, *_):
        return self

    def eq(self, key, value):
        self.query[key] = value
        return self

    def limit(self, _):
        return self

    def update(self, values):
        for row in self.db.get(self.name, []):
            if all(row.get(k) == v for k, v in self.query.items()):
                row.update(values)
        return self

    def insert(self, rows):
        if self.name not in self.db:
            self.db[self.name] = []
        self.db[self.name].extend(rows)
        return self

    def execute(self):
        if self.query:
            filtered = [
                row for row in self.db.get(self.name, [])
                if all(row.get(k) == v for k, v in self.query.items())
            ]
            return MockResponse(filtered)
        return MockResponse(self.db.get(self.name, []))


class MockSupabase:
    def __init__(self, db):
        self.db = db

    def table(self, name):
        return MockTable(name, self.db)


# -----------------------
# Mock Model
# -----------------------

class MockModel:
    def encode(self, texts, convert_to_numpy=True):
        # deterministic fake embeddings based on text length
        return np.array([[len(str(t))] for t in texts], dtype=float)


# -----------------------
# TEST: Happy Path
# -----------------------

def test_happy_path(monkeypatch):

    db = {
        "resumes": [
            {"id": "i1", "term": "Summer 2026", "text": "python machine learning"},
            {"id": "i2", "term": "Summer 2026", "text": "frontend react"}
        ],
        "projects": [
            {"id": "p1", "term": "Summer 2026", "description": "ml project", "deliverable": "", "requirements": ""},
            {"id": "p2", "term": "Summer 2026", "description": "react ui", "deliverable": "", "requirements": ""}
        ],
        "results_tab": [],
    }

    mock_supabase = MockSupabase(db)
    mock_model = MockModel()

    monkeypatch.setattr("draft_2.supabase", mock_supabase)
    monkeypatch.setattr("draft_2.model", mock_model)

    result = run_matching("Summer 2026", "job-1")

    assert result is True
    assert len(db["results_tab"]) == 2

    row = db["results_tab"][0]
    assert "intern_id" in row
    assert "projects" in row
    assert len(row["projects"]) > 0
    assert row["recommended_project_id"] is not None


# -----------------------
# TEST: No Jobs in Queue
# -----------------------

def test_no_jobs_in_queue(monkeypatch):

    db = {
        "jobs": []
    }

    mock_supabase = MockSupabase(db)

    monkeypatch.setattr("draft_2.supabase", mock_supabase)

    jobs = (
        mock_supabase.table("jobs")
        .select("*")
        .eq("status", "ready")
        .limit(1)
        .execute()
    )

    assert jobs.data == []


# -----------------------
# TEST: Empty Interns
# -----------------------

def test_empty_interns(monkeypatch):

    db = {
        "resumes": [],
        "projects": [
            {"id": "p1", "term": "Summer 2026", "description": "ml", "deliverable": "", "requirements": ""}
        ],
        "results_tab": [],
    }

    mock_supabase = MockSupabase(db)
    mock_model = MockModel()

    monkeypatch.setattr("draft_2.supabase", mock_supabase)
    monkeypatch.setattr("draft_2.model", mock_model)

    with pytest.raises(Exception) as exc:
        run_matching("Summer 2026", "job-1")

    assert "No interns or projects found" in str(exc.value)


# -----------------------
# TEST: Empty Projects
# -----------------------

def test_empty_projects(monkeypatch):

    db = {
        "resumes": [
            {"id": "i1", "term": "Summer 2026", "text": "python"}
        ],
        "projects": [],
        "results_tab": [],
    }

    mock_supabase = MockSupabase(db)
    mock_model = MockModel()

    monkeypatch.setattr("draft_2.supabase", mock_supabase)
    monkeypatch.setattr("draft_2.model", mock_model)

    with pytest.raises(Exception):
        run_matching("Summer 2026", "job-1")