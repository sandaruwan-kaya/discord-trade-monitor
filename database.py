import sqlite3
import json
from contextlib import closing
from datetime import datetime, timezone

import config


#
# Absolute path, resolved from the project directory.
# Kept as a module attribute for backwards compatibility;
# get_connection() re-resolves so tests can override it.
#

DB_FILE = config.db_path()

SCHEMA_VERSION = 1


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_connection():
    return sqlite3.connect(
        config.db_path(),
        check_same_thread=False
    )


#
# Columns added to the original signals table in schema v1.
# ALTER TABLE ADD COLUMN only; existing rows are untouched.
#

SIGNAL_COLUMNS_V1 = {
    "timeframe": "TEXT",
    "raw_message": "TEXT",
    "created_at": "TEXT",
    "updated_at": "TEXT",
    "last_processed_candle_ms": "INTEGER",
    "eval_state": "TEXT",
    "target_flags": "TEXT",
    "first_working_at": "TEXT",
}


def initialize_database():
    with closing(get_connection()) as conn:

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                discord_message_id TEXT UNIQUE,

                symbol TEXT NOT NULL,
                bias TEXT,
                summary TEXT,

                signal_time_ms INTEGER NOT NULL,
                price_at_signal REAL,

                support_levels TEXT,
                resistance_levels TEXT,
                targets TEXT,

                confirmation_above REAL,
                confirmation_below REAL,

                invalidation_above REAL,
                invalidation_below REAL,

                status TEXT DEFAULT 'DEVELOPING',

                active INTEGER DEFAULT 1
            )
            """
        )

        conn.commit()

        migrate(conn)


def migrate(conn):
    version = conn.execute("PRAGMA user_version").fetchone()[0]

    if version >= SCHEMA_VERSION:
        return

    with conn:
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(signals)")
        }

        for name, sql_type in SIGNAL_COLUMNS_V1.items():
            if name not in existing:
                conn.execute(
                    f"ALTER TABLE signals ADD COLUMN {name} {sql_type}"
                )

        #
        # Backfill timestamps for rows created before v1.
        #

        conn.execute(
            """
            UPDATE signals
            SET created_at = strftime(
                    '%Y-%m-%dT%H:%M:%S+00:00',
                    signal_time_ms / 1000, 'unixepoch'
                )
            WHERE created_at IS NULL
            """
        )

        conn.execute(
            """
            UPDATE signals
            SET updated_at = created_at
            WHERE updated_at IS NULL
            """
        )

        #
        # Audit trail of signal status changes.
        #

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS signal_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER NOT NULL REFERENCES signals(id),
                created_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                old_status TEXT,
                new_status TEXT,
                details TEXT
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_signal_events_signal
            ON signal_events(signal_id)
            """
        )

        #
        # Agent 1 output, one row per analysis run.
        #

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS signal_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER NOT NULL REFERENCES signals(id),
                created_at TEXT NOT NULL,
                signal_state TEXT NOT NULL,
                next_step TEXT NOT NULL,
                llm_used INTEGER NOT NULL,
                input_json TEXT NOT NULL,
                output_json TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_signal_analyses_signal
            ON signal_analyses(signal_id)
            """
        )

        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def save_signal(
    discord_message_id,
    signal,
    signal_time_ms,
    price_at_signal,
    raw_message=None,
    target_flags=None
):
    now = utc_now_iso()

    with closing(get_connection()) as conn:

        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO signals (
                discord_message_id,
                symbol,
                bias,
                summary,
                signal_time_ms,
                price_at_signal,
                support_levels,
                resistance_levels,
                targets,
                confirmation_above,
                confirmation_below,
                invalidation_above,
                invalidation_below,
                status,
                active,
                timeframe,
                raw_message,
                target_flags,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
            """,
            (
                str(discord_message_id),
                signal.symbol,
                signal.bias,
                signal.summary,
                signal_time_ms,
                price_at_signal,
                json.dumps(signal.support_levels),
                json.dumps(signal.resistance_levels),
                json.dumps(signal.targets),
                signal.confirmation_above,
                signal.confirmation_below,
                signal.invalidation_above,
                signal.invalidation_below,
                "DEVELOPING",
                signal.timeframe,
                raw_message,
                json.dumps(target_flags) if target_flags else None,
                now,
                now
            )
        )

        inserted = cursor.rowcount > 0

        conn.commit()

    return inserted


def _fetch(sql, params=()):
    with closing(get_connection()) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()

    return [
        dict(row)
        for row in rows
    ]


def get_active_signals():
    return _fetch(
        """
        SELECT *
        FROM signals
        WHERE active = 1
        ORDER BY id ASC
        """
    )


def get_signal(signal_id):
    rows = _fetch("SELECT * FROM signals WHERE id = ?", (signal_id,))
    return rows[0] if rows else None


def get_signal_by_discord_message_id(discord_message_id):
    rows = _fetch(
        "SELECT * FROM signals WHERE discord_message_id = ?",
        (str(discord_message_id),)
    )
    return rows[0] if rows else None


def update_status(
    signal_id,
    status,
    active=1
):
    now = utc_now_iso()

    with closing(get_connection()) as conn:

        conn.execute(
            """
            UPDATE signals
            SET status = ?,
                active = ?,
                updated_at = ?,
                first_working_at = CASE
                    WHEN ? = 'WORKING' AND first_working_at IS NULL
                    THEN ?
                    ELSE first_working_at
                END
            WHERE id = ?
            """,
            (
                status,
                active,
                now,
                status,
                now,
                signal_id
            )
        )

        conn.commit()


def update_eval_state(
    signal_id,
    state,
    last_processed_candle_ms
):
    with closing(get_connection()) as conn:

        conn.execute(
            """
            UPDATE signals
            SET eval_state = ?,
                last_processed_candle_ms = ?
            WHERE id = ?
            """,
            (
                json.dumps(state),
                last_processed_candle_ms,
                signal_id
            )
        )

        conn.commit()


def record_signal_event(
    signal_id,
    event_type,
    old_status=None,
    new_status=None,
    details=None
):
    with closing(get_connection()) as conn:

        conn.execute(
            """
            INSERT INTO signal_events (
                signal_id, created_at, event_type,
                old_status, new_status, details
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                signal_id,
                utc_now_iso(),
                event_type,
                old_status,
                new_status,
                json.dumps(details, default=str) if details is not None else None
            )
        )

        conn.commit()


def get_signal_events(signal_id):
    return _fetch(
        "SELECT * FROM signal_events WHERE signal_id = ? ORDER BY id",
        (signal_id,)
    )


def save_signal_analysis(
    signal_id,
    analysis_input,
    analysis
):
    """Persist one Agent 1 run. Both args are Pydantic models."""
    with closing(get_connection()) as conn:

        cursor = conn.execute(
            """
            INSERT INTO signal_analyses (
                signal_id, created_at, signal_state, next_step,
                llm_used, input_json, output_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal_id,
                utc_now_iso(),
                analysis.signal_state,
                analysis.next_step,
                1 if analysis.llm_used else 0,
                analysis_input.model_dump_json(),
                analysis.model_dump_json()
            )
        )

        conn.commit()

    return cursor.lastrowid


def get_signal_analyses(signal_id):
    return _fetch(
        "SELECT * FROM signal_analyses WHERE signal_id = ? ORDER BY id",
        (signal_id,)
    )
