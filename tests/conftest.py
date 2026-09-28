"""
Test harness guarantees:

- dummy env values are set BEFORE project modules load, so real
  .env secrets are never used (load_dotenv does not override)
- every outbound socket connection raises (no Discord, Telegram,
  OpenAI, Binance or TradingView calls can happen)
- each test gets its own temporary SQLite database
"""
import os
import socket

import pytest


_TEST_ENV = {
    "DISCORD_BOT_TOKEN": "test-discord-token",
    "DISCORD_CHANNEL_ID": "123",
    "OPENAI_API_KEY": "test-openai-key",
    "OPENAI_MODEL": "test-model",
    "TELEGRAM_BOT_TOKEN": "test-telegram-token",
    "TELEGRAM_CHAT_ID": "0",
    "AGENT1_ENABLED": "false",
}

os.environ.update(_TEST_ENV)


class NetworkBlocked(RuntimeError):
    pass


def _blocked(*args, **kwargs):
    raise NetworkBlocked("network access is disabled in tests")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    monkeypatch.setenv("TRADE_MONITOR_DB", str(path))

    import config
    import database
    database.initialize_database()

    yield path

    # A test that undid the DB override would have pointed at the real file.
    assert config.db_path() != str(config.PROJECT_DIR / "trade_monitor.db"), \
        "test ended pointing at the real project database"
