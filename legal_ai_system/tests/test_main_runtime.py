import os
import sys
import time
import subprocess
from pathlib import Path


def test_main_runs_without_errors():
    """Ensure main.py starts without immediate runtime errors."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "main.py"
    env = os.environ.copy()
    env.setdefault("LEGAL_AI_API_PORT", "8123")
    env.setdefault("LEGAL_AI_API_HOST", "127.0.0.1")

    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    try:
        time.sleep(3)
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            output = stdout.decode() + stderr.decode()
            assert proc.returncode == 0, f"main.py exited with code {proc.returncode}\n{output}"
            assert "Traceback" not in output, f"Exception encountered:\n{output}"
        else:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    finally:
        if proc.poll() is None:
            proc.kill()
