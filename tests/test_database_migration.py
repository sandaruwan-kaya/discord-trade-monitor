"""Additive schema migration of the original signals table."""
import sqlite3

import database


LEGACY_SCHEMA = """
CREATE TABLE signals (
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


def make_legacy_db(path):
    conn = sqlite3.connect(path)
    conn.execute(LEGACY_SCHEMA)
    conn.execute(
        """
        INSERT INTO signals (discord_message_id, symbol, bias, summary,
            signal_time_ms, price_at_signal, support_levels, resistance_levels,
            targets, confirmation_above, invalidation_below, status, active)
        VALUES ('555', 'BTCUSDT', 'conditional_bullish', 'legacy',
            1790363633126, 84057.55, '[]', '[]', '[84300.0]', 84050.0, 83750.0,
            'WORKING', 1)
        """
    )
    conn.commit()
    conn.close()


def columns(path):
    conn = sqlite3.connect(path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(signals)")}
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    return cols, tables, version


def test_migrates_legacy_db_preserving_rows(tmp_path, monkeypatch):
    path = tmp_path / "legacy.db"
    make_legacy_db(path)
    monkeypatch.setenv("TRADE_MONITOR_DB", str(path))

    database.initialize_database()

    cols, tables, version = columns(path)
    assert set(database.SIGNAL_COLUMNS_V1) <= cols
    assert {"signal_events", "signal_analyses", "trade_proposals"} <= tables
    assert version == database.SCHEMA_VERSION

    rows = database.get_active_signals()
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "WORKING"
    assert row["price_at_signal"] == 84057.55
    assert row["created_at"] == "2026-09-25T19:13:53+00:00"
    assert row["updated_at"] == row["created_at"]
    assert row["last_processed_candle_ms"] is None


def test_migration_is_idempotent(tmp_path, monkeypatch):
    path = tmp_path / "legacy.db"
    make_legacy_db(path)
    monkeypatch.setenv("TRADE_MONITOR_DB", str(path))

    database.initialize_database()
    database.initialize_database()

    assert len(database.get_active_signals()) == 1


def test_update_status_sets_first_working_at_once():
    from helpers import bullish_signal

    database.save_signal("1", bullish_signal(), 1, 100.0)
    signal_id = database.get_signal_by_discord_message_id("1")["id"]

    database.update_status(signal_id, "WORKING")
    first = database.get_signal(signal_id)["first_working_at"]
    assert first is not None

    database.update_status(signal_id, "DEVELOPING")
    database.update_status(signal_id, "WORKING")
    assert database.get_signal(signal_id)["first_working_at"] == first


def test_db_path_is_absolute_and_project_relative(monkeypatch):
    import config

    with monkeypatch.context() as m:
        m.delenv("TRADE_MONITOR_DB")
        path = config.db_path()
    assert path == str(config.PROJECT_DIR / "trade_monitor.db")
    assert path.startswith("/")
