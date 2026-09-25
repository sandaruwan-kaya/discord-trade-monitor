import sqlite3
import json


DB_FILE = "trade_monitor.db"


def get_connection():
    return sqlite3.connect(
        DB_FILE,
        check_same_thread=False
    )


def initialize_database():
    conn = get_connection()

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
    conn.close()


def save_signal(
    discord_message_id,
    signal,
    signal_time_ms,
    price_at_signal
):
    conn = get_connection()

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
            active
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
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
            "DEVELOPING"
        )
    )

    inserted = cursor.rowcount > 0

    conn.commit()
    conn.close()

    return inserted


def get_active_signals():
    conn = get_connection()
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT *
        FROM signals
        WHERE active = 1
        ORDER BY id ASC
        """
    ).fetchall()

    conn.close()

    return [
        dict(row)
        for row in rows
    ]


def update_status(
    signal_id,
    status,
    active=1
):
    conn = get_connection()

    conn.execute(
        """
        UPDATE signals
        SET status = ?,
            active = ?
        WHERE id = ?
        """,
        (
            status,
            active,
            signal_id
        )
    )

    conn.commit()
    conn.close()