"""Live smoke script: hits real services. Never collected by pytest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telegram_client import send_telegram


def main():
    print("Sending Telegram test message...")

    result = send_telegram(
        "✅ Telegram integration from EC2 is working."
    )

    print("Telegram API response:")
    print(result)


if __name__ == "__main__":
    main()
