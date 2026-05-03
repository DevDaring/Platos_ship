import os
import sys
import atexit
import signal
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the same directory as this file (project root)
load_dotenv(Path(__file__).parent / ".env", override=False)

PHONE = os.getenv("PHONE_NO", "")
API_KEY = os.getenv("TextBelt_API_KEY", "")

# Internal flag: True once any terminal notification has been sent,
# so the atexit handler never sends a duplicate.
_notified = False


def send_sms(msg: str) -> bool:
    """Send an SMS via TextBelt. Returns True on success."""
    global _notified
    if not PHONE or not API_KEY:
        print("[Notifier] PHONE_NO or TextBelt_API_KEY not set — skipping SMS")
        return False
    try:
        r = requests.post(
            "https://textbelt.com/text",
            data={"phone": PHONE, "message": msg, "key": API_KEY},
            timeout=20,
        )
        data = r.json()
        if data.get("success"):
            print(f"[Notifier] SMS sent. Quota left: {data.get('quotaRemaining')}")
            _notified = True
            return True
        else:
            print(f"[Notifier] TextBelt error: {data.get('error')}")
            return False
    except Exception as e:
        print(f"[Notifier] Failed to send SMS: {e}")
        return False


def _atexit_handler():
    """Fire only for unexpected exits (process killed, uncaught signal, etc.)."""
    if not _notified:
        send_sms("⚠️ Plato's Ship: process exited without a normal completion message")


def register_exit_notifier():
    """
    Register atexit and SIGTERM handler so the user is notified even if the
    pipeline dies unexpectedly.  Call this once from main() after load_dotenv().

    SIGINT (Ctrl-C) is intentionally NOT registered here — the TrialRunner
    owns SIGINT so it can flush the checkpoint first, then calls send_sms().

    The atexit handler fires only when no explicit send_sms() call has been
    made yet, preventing duplicate messages.
    """
    atexit.register(_atexit_handler)

    def on_sigterm(signum, frame):
        send_sms("\u26a0\ufe0f Plato's Ship: terminated by SIGTERM")
        sys.exit(0)

    signal.signal(signal.SIGTERM, on_sigterm)