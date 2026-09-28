"""
Persisted, idempotent trade-event notifications.

    persist state first -> notify(key, ...) second

notification_key is UNIQUE. A key already SENT is never sent again (not
after a restart either). A failed send never touches trading state; it is
retried when the same key comes up again, at most once per
NOTIFY_RETRY_SECONDS and up to MAX_ATTEMPTS, so a short Telegram outage
cannot exhaust the attempts inside a single cycle.
Message texts contain identifiers and prices only, never secrets.
"""
import sqlite3
import time
from contextlib import closing

import config
import database


MAX_ATTEMPTS = 20


class Notifier:

    def __init__(self, send, log=print, now_ms=None):
        self._send = send
        self._log = log
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))

    def notify(self, key: str, kind: str, text: str) -> bool:
        """Returns True if this call delivered the message."""
        now = database.utc_now_iso()
        with closing(database.get_connection()) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute(
                """
                INSERT OR IGNORE INTO notifications (notification_key, kind, text, status, created_at)
                VALUES (?, ?, ?, 'PENDING', ?)
                """,
                (key, kind, text, now)
            )
            conn.commit()
            row = conn.execute("SELECT * FROM notifications WHERE notification_key = ?", (key,)).fetchone()

        if row["status"] == "SENT" or row["attempts"] >= MAX_ATTEMPTS:
            return False
        now_ms = self._now_ms()
        if row["last_attempt_at_ms"] is not None and \
                now_ms - row["last_attempt_at_ms"] < config.notify_retry_seconds() * 1000:
            return False

        try:
            self._send(row["text"])
            status, error = "SENT", None
        except Exception as e:
            status, error = "FAILED", f"{type(e).__name__}: {str(e)[:200]}"
            self._log(f"[NOTIFY] {key}: send failed ({type(e).__name__}); state unaffected")

        with closing(database.get_connection()) as conn:
            conn.execute(
                """
                UPDATE notifications SET status = ?, attempts = attempts + 1, last_error = ?,
                    last_attempt_at_ms = ?, sent_at = CASE WHEN ? = 'SENT' THEN ? ELSE sent_at END
                WHERE notification_key = ? AND status != 'SENT'
                """,
                (status, error, now_ms, status, database.utc_now_iso(), key)
            )
            conn.commit()

        return status == "SENT"
