"""
Vercel Serverless Function entry point for FastAPI backend.
"""
import os
import sys
import shutil
from pathlib import Path

# Mark Vercel runtime
os.environ["VERCEL"] = "1"
os.environ["ENVIRONMENT"] = "production"

root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"

if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Ensure writable /tmp directories on Vercel
tmp_data_raw = Path("/tmp/data/raw")
tmp_data_processed = Path("/tmp/data/processed")
tmp_data_sample = Path("/tmp/data/sample")
tmp_models = Path("/tmp/models")

for p in (tmp_data_raw, tmp_data_processed, tmp_data_sample, tmp_models):
    p.mkdir(parents=True, exist_ok=True)

# Copy initial CSV files from backend/data/raw to /tmp/data/raw
source_raw = backend_dir / "data" / "raw"
if source_raw.exists():
    for f in source_raw.glob("*.csv"):
        dest_f = tmp_data_raw / f.name
        if not dest_f.exists():
            try:
                shutil.copy2(f, dest_f)
            except Exception:
                pass

from app.main import app
