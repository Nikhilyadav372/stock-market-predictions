"""
Rock-solid launcher to start FastAPI backend and Cloudflare Tunnel together.
Prints your live public HTTPS URL, keeps it alive, and displays live API requests.
"""
import os
import re
import sys
import time
import subprocess
import threading

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

def log(msg=""):
    print(msg, flush=True)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
VENV_PYTHON = os.path.join(BACKEND_DIR, "venv", "Scripts", "python.exe")
CLOUDFLARED_EXE = os.path.join(ROOT_DIR, "cloudflared.exe")

if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = sys.executable

if not os.path.exists(CLOUDFLARED_EXE):
    log("[ERROR] cloudflared.exe not found in project root.")
    sys.exit(1)

log("=" * 65)
log(">>> STARTING AI STOCK FORECASTING BACKEND & CLOUDFLARE TUNNEL <<<")
log("=" * 65)

# 1. Start FastAPI Backend
log("\n[1/2] Starting FastAPI Backend on http://127.0.0.1:8000...")
backend_proc = subprocess.Popen(
    [VENV_PYTHON, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
    cwd=BACKEND_DIR,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",
    errors="replace",
    bufsize=1,
)

# Stream backend logs in background
def stream_backend_logs():
    try:
        for line in iter(backend_proc.stdout.readline, ""):
            line_str = line.strip()
            # Only display interesting API requests or errors
            if "INFO:" in line_str or "ERROR:" in line_str or "WARNING:" in line_str:
                if "/api/" in line_str or "docs" in line_str or "ERROR" in line_str:
                    log(f"   [API Request] {line_str}")
    except Exception:
        pass

t = threading.Thread(target=stream_backend_logs, daemon=True)
t.start()

time.sleep(2)

# 2. Start Cloudflare Tunnel
log("[2/2] Launching Cloudflare HTTPS Tunnel...")
tunnel_proc = subprocess.Popen(
    [CLOUDFLARED_EXE, "tunnel", "--url", "http://127.0.0.1:8000"],
    cwd=ROOT_DIR,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",
    errors="replace",
    bufsize=1,
)

public_url = None
url_regex = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

try:
    for line in iter(tunnel_proc.stdout.readline, ""):
        match = url_regex.search(line)
        if match:
            public_url = match.group(0)
            break

    if public_url:
        try:
            subprocess.run("clip", input=public_url.strip(), text=True, check=False)
            copied_msg = " [COPIED TO CLIPBOARD! Press Ctrl+V in Vercel]"
        except Exception:
            copied_msg = ""

        log("\n" + "=" * 65)
        log(">>> SUCCESS! YOUR BACKEND IS LIVE ON THE INTERNET (HTTPS)! <<<")
        log("=" * 65)
        log(f"\n[+] PUBLIC BACKEND URL:\n    {public_url}")
        log(f"[+] SWAGGER DOCS:\n    {public_url}/docs\n")
        log(f"[*] Note:{copied_msg}")
        log("\n" + "-" * 65)
        log("IMPORTANT - KYU 'SOMETHING WENT WRONG' AATA HAI:")
        log("1. Ye black window hamesha OPEN rehni chahiye jab website use kar rahe ho.")
        log("2. Agar laptop sleep ho gaya ya window close ho gayi, tunnel band ho jata hai.")
        log(f"3. Vercel mein VITE_API_BASE_URL check karein ki wo yahi URL ho:")
        log(f"   {public_url}")
        log("-" * 65)
        log("\n[!] LIVE INCOMING REQUESTS FROM VERCEL WILL APPEAR BELOW:")
        log("    (Press Ctrl + C anytime to stop)\n")

        tunnel_proc.wait()
    else:
        log("[!] Could not retrieve Cloudflare Tunnel URL. Check logs.")

except KeyboardInterrupt:
    log("\n[!] Shutting down backend and tunnel...")
finally:
    try:
        backend_proc.terminate()
        tunnel_proc.terminate()
    except Exception:
        pass
    log("[OK] All processes stopped cleanly.")
