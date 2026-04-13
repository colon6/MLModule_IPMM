import os
import psycopg

DATABASE_URL = os.environ.get("DATABASE_URL") if not DATABASE_URL: raise SystemExit("Set DATABASE_URL environment variable")