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
