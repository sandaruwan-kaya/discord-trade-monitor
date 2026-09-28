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

SCHEMA_VERSION = 4


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

    if version < 1:
        _migrate_v1(conn)

    if version < 2:
        _migrate_v2(conn)

    if version < 3:
        _migrate_v3(conn)

    if version < 4:
        _migrate_v4(conn)


def _migrate_v1(conn):
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

        conn.execute("PRAGMA user_version = 1")


def _migrate_v2(conn):
    with conn:
        #
        # Agent 2 proposals. proposal_key is UNIQUE: claiming a key
        # with INSERT OR IGNORE is what makes a proposal exactly-once.
        #

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_proposals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                proposal_key TEXT NOT NULL UNIQUE,
                signal_id INTEGER NOT NULL REFERENCES signals(id),
                status TEXT NOT NULL,
                trigger TEXT,
                strategy TEXT,
                contract_symbol TEXT,
                option_type TEXT,
                expiry TEXT,
                strike REAL,
                limit_price REAL,
                quantity INTEGER,
                contract_multiplier REAL,
                estimated_max_loss REAL,
                reason TEXT,
                error TEXT,
                input_json TEXT,
                output_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_trade_proposals_signal
            ON trade_proposals(signal_id)
            """
        )

        conn.execute("PRAGMA user_version = 2")


def _migrate_v3(conn):
    """
    Paper broker state. CHECK constraints make illegal states
    unrepresentable: only BUY_TO_OPEN / SELL_TO_CLOSE, positive order
    quantities, no negative (short) position quantities. UNIQUE keys
    give idempotency (client_order_id), one fill per order (no partial
    fills in V1) and one position per filled entry order.
    """
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_orders (
                order_id TEXT PRIMARY KEY,
                client_order_id TEXT NOT NULL UNIQUE,
                signal_id INTEGER,
                proposal_key TEXT,
                position_id TEXT,
                contract_symbol TEXT NOT NULL,
                side TEXT NOT NULL
                    CHECK (side IN ('BUY_TO_OPEN', 'SELL_TO_CLOSE')),
                quantity INTEGER NOT NULL CHECK (quantity > 0),
                limit_price REAL NOT NULL CHECK (limit_price > 0),
                status TEXT NOT NULL
                    CHECK (status IN ('PENDING', 'FILLED', 'CANCELLED', 'REJECTED')),
                fill_price REAL,
                rejection_reason TEXT,
                last_quote_bid REAL,
                last_quote_ask REAL,
                last_checked_at TEXT,
                submitted_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                filled_at TEXT,
                cancelled_at TEXT
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_fills (
                fill_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL UNIQUE REFERENCES paper_orders(order_id),
                contract_symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL CHECK (quantity > 0),
                price REAL NOT NULL CHECK (price > 0),
                quote_bid REAL,
                quote_ask REAL,
                filled_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_positions (
                position_id TEXT PRIMARY KEY,
                entry_order_id TEXT NOT NULL UNIQUE REFERENCES paper_orders(order_id),
                signal_id INTEGER,
                proposal_key TEXT,
                contract_symbol TEXT NOT NULL,
                underlying TEXT,
                option_type TEXT,
                contract_multiplier REAL NOT NULL CHECK (contract_multiplier > 0),
                quantity INTEGER NOT NULL CHECK (quantity >= 0),
                original_quantity INTEGER NOT NULL CHECK (original_quantity > 0),
                entry_price REAL NOT NULL CHECK (entry_price > 0),
                current_mark REAL,
                mark_as_of TEXT,
                unrealized_pnl REAL NOT NULL DEFAULT 0,
                realized_pnl REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL CHECK (status IN ('OPEN', 'CLOSED')),
                opened_at TEXT NOT NULL,
                closed_at TEXT,
                updated_at TEXT NOT NULL,
                CHECK (status = 'CLOSED' OR quantity > 0)
            )
            """
        )

        for sql in (
            "CREATE INDEX IF NOT EXISTS idx_paper_orders_status ON paper_orders(status)",
            "CREATE INDEX IF NOT EXISTS idx_paper_orders_signal ON paper_orders(signal_id)",
            "CREATE INDEX IF NOT EXISTS idx_paper_positions_status ON paper_positions(status)",
            "CREATE INDEX IF NOT EXISTS idx_paper_positions_signal ON paper_positions(signal_id)",
        ):
            conn.execute(sql)

        conn.execute("PRAGMA user_version = 3")


def _migrate_v4(conn):
    """
    Policy decisions audit. Each row stores the exact input snapshot and
    the exact limits used, so later config changes never rewrite history.
    evaluation_key (hash of the input snapshot) is UNIQUE: re-running the
    same snapshot never duplicates a row; a newer snapshot is a new row.
    """
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS policy_decisions (
                decision_id TEXT PRIMARY KEY,
                evaluation_key TEXT NOT NULL UNIQUE,
                proposal_key TEXT NOT NULL,
                signal_id INTEGER,
                purpose TEXT NOT NULL,
                approved INTEGER NOT NULL CHECK (approved IN (0, 1)),
                decision TEXT NOT NULL CHECK (decision IN ('APPROVED', 'REJECTED')),
                reasons_json TEXT NOT NULL,
                calculated_max_loss REAL,
                calculated_total_open_risk REAL,
                quote_age_seconds REAL,
                checked_at TEXT NOT NULL,
                input_json TEXT NOT NULL,
                policy_snapshot_json TEXT NOT NULL,
                decision_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                CHECK ((approved = 1) = (decision = 'APPROVED'))
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_policy_decisions_proposal
            ON policy_decisions(proposal_key)
            """
        )

        conn.execute("PRAGMA user_version = 4")


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


def get_latest_signal_analysis(signal_id):
    rows = _fetch(
        "SELECT * FROM signal_analyses WHERE signal_id = ? ORDER BY id DESC LIMIT 1",
        (signal_id,)
    )
    return rows[0] if rows else None


#
# Trade proposals (Agent 2)
#

def get_signals_awaiting_proposal():
    """Active WORKING signals that reached WORKING and have no proposal row yet."""
    return _fetch(
        """
        SELECT s.*
        FROM signals s
        WHERE s.active = 1
          AND s.status = 'WORKING'
          AND s.first_working_at IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM trade_proposals p
              WHERE p.signal_id = s.id
          )
        ORDER BY s.id ASC
        """
    )


def claim_proposal(proposal_key, signal_id):
    """
    Atomically claim the proposal slot. True only for the single
    caller that inserted the row; every later call gets False.
    """
    now = utc_now_iso()

    with closing(get_connection()) as conn:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO trade_proposals (
                proposal_key, signal_id, status, created_at, updated_at
            )
            VALUES (?, ?, 'PENDING', ?, ?)
            """,
            (proposal_key, signal_id, now, now)
        )
        conn.commit()

    return cursor.rowcount == 1


def finish_proposal(
    proposal_key,
    status,
    proposal=None,
    selection_input=None,
    reason=None,
    error=None,
    trigger=None
):
    """Move a PENDING claim to its final status. Never touches non-PENDING rows."""
    p = proposal

    with closing(get_connection()) as conn:
        cursor = conn.execute(
            """
            UPDATE trade_proposals
            SET status = ?,
                trigger = ?,
                strategy = ?,
                contract_symbol = ?,
                option_type = ?,
                expiry = ?,
                strike = ?,
                limit_price = ?,
                quantity = ?,
                contract_multiplier = ?,
                estimated_max_loss = ?,
                reason = ?,
                error = ?,
                input_json = ?,
                output_json = ?,
                updated_at = ?
            WHERE proposal_key = ? AND status = 'PENDING'
            """,
            (
                status,
                (p.trigger if p else trigger),
                p.strategy if p else None,
                p.contract_symbol if p else None,
                p.option_type if p else None,
                str(p.expiry) if p and p.expiry else None,
                p.strike if p else None,
                p.limit_price if p else None,
                p.quantity if p else None,
                p.contract_multiplier if p else None,
                p.estimated_max_loss if p else None,
                (p.reason if p else reason),
                error,
                selection_input.model_dump_json() if selection_input else None,
                p.model_dump_json() if p else None,
                utc_now_iso(),
                proposal_key
            )
        )
        conn.commit()

    return cursor.rowcount == 1


def fail_interrupted_proposals(older_than_iso):
    """
    PENDING claims created before older_than_iso were interrupted
    (crash / restart). Mark FAILED; they are never retried.
    Returns the affected proposal keys.
    """
    with closing(get_connection()) as conn:
        keys = [
            row[0] for row in conn.execute(
                "SELECT proposal_key FROM trade_proposals WHERE status = 'PENDING' AND created_at < ?",
                (older_than_iso,)
            )
        ]

        conn.execute(
            """
            UPDATE trade_proposals
            SET status = 'FAILED',
                error = 'interrupted: PENDING claim was never completed; not retried',
                updated_at = ?
            WHERE status = 'PENDING' AND created_at < ?
            """,
            (utc_now_iso(), older_than_iso)
        )
        conn.commit()

    return keys


def get_proposal(proposal_key):
    rows = _fetch("SELECT * FROM trade_proposals WHERE proposal_key = ?", (proposal_key,))
    return rows[0] if rows else None


def get_proposals_for_signal(signal_id):
    return _fetch(
        "SELECT * FROM trade_proposals WHERE signal_id = ? ORDER BY id",
        (signal_id,)
    )
