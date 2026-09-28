"""The test harness itself: no network, no Discord start."""
import socket

import pytest


def test_network_is_blocked():
    with pytest.raises(RuntimeError):
        socket.create_connection(("api.telegram.org", 443), timeout=1)


def test_requests_cannot_reach_telegram():
    import telegram_client

    with pytest.raises(Exception):
        telegram_client.send_telegram("must never be sent")


def test_importing_app_does_not_start_discord():
    import app

    assert app.client.is_closed() is False
    assert app.client.user is None                  # never logged in
    assert app.services.signal_agent is not None


def test_real_env_secrets_not_loaded():
    import os

    assert os.environ["OPENAI_API_KEY"] == "test-openai-key"
    assert os.environ["DISCORD_BOT_TOKEN"] == "test-discord-token"
