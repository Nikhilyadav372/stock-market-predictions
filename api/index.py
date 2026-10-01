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
tmp_data = Path("/tmp/data")
tmp_data_raw = Path("/tmp/data/raw")
tmp_data_processed = Path("/tmp/data/processed")
tmp_data_sample = Path("/tmp/data/sample")
tmp_models = Path("/tmp/models")

for p in (tmp_data, tmp_data_raw, tmp_data_processed, tmp_data_sample, tmp_models):
    p.mkdir(parents=True, exist_ok=True)

# 1. Seed SQLite Database from backend/data/seed.db or dev.db if not in /tmp
tmp_db = Path("/tmp/dev.db")
if not tmp_db.exists() or tmp_db.stat().st_size == 0:
    for candidate in [backend_dir / "data" / "seed.db", backend_dir / "data" / "dev.db"]:
        if candidate.exists() and candidate.stat().st_size > 0:
            try:
                shutil.copy2(candidate, tmp_db)
                break
            except Exception:
                pass

# 2. Copy raw CSV files to /tmp/data/raw
source_raw = backend_dir / "data" / "raw"
if source_raw.exists():
    for f in source_raw.glob("*.csv"):
        dest_f = tmp_data_raw / f.name
        if not dest_f.exists() or dest_f.stat().st_size == 0:
            try:
                shutil.copy2(f, dest_f)
            except Exception:
                pass

# 3. Ensure database schema is created unconditionally (not depending on lifespan)
try:
    from app.database import engine, Base, SessionLocal
    from app.models import Stock, HistoricalPrice
    import pandas as pd

    Base.metadata.create_all(bind=engine)

    # If stocks table is empty, auto-seed from CSV files
    db = SessionLocal()
    try:
        stock_count = db.query(Stock).count()
        if stock_count == 0:
            POPULAR_NAMES = {
                "AAPL": "Apple Inc.",
                "MSFT": "Microsoft Corp.",
                "NVDA": "NVIDIA Corp.",
                "TSLA": "Tesla Inc.",
                "JPM": "JPMorgan Chase & Co.",
                "RELIANCE.NS": "Reliance Industries",
                "TCS.NS": "Tata Consultancy Services",
            }
            for csv_file in tmp_data_raw.glob("*.csv"):
                symbol = csv_file.stem.upper()
                name = POPULAR_NAMES.get(symbol, symbol)
                stock = Stock(symbol=symbol, name=name, currency="USD" if not symbol.endswith(".NS") else "INR")
                db.add(stock)
                db.flush()

                try:
                    df = pd.read_csv(csv_file, index_col="Date")
                    for idx, row in df.iterrows():
                        d_val = pd.to_datetime(idx).date()
                        hp = HistoricalPrice(
                            stock_id=stock.id,
                            date=d_val,
                            open=float(row["Open"]),
                            high=float(row["High"]),
                            low=float(row["Low"]),
                            close=float(row["Close"]),
                            volume=int(row["Volume"]),
                            adj_close=float(row.get("Adj_Close", row["Close"])),
                        )
                        db.add(hp)
                    db.commit()
                except Exception:
                    db.rollback()
    finally:
        db.close()
except Exception as e:
    print(f"Warning during DB bootstrap: {e}")

# 4. Import FastAPI app
from app.main import app
