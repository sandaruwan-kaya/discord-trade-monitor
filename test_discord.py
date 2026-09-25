import os
import discord
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("DISCORD_BOT_TOKEN")
EXPECTED_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID"))

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print("=" * 60)
    print(f"CONNECTED AS: {client.user}")
    print(f"EXPECTED CHANNEL ID: {EXPECTED_CHANNEL_ID}")
    print("=" * 60)

    print("\nServers visible to bot:")

    for guild in client.guilds:
        print(f"\nSERVER: {guild.name}")
        print(f"SERVER ID: {guild.id}")

        for channel in guild.text_channels:
            print(
                f"  #{channel.name} "
                f"| ID={channel.id} "
                f"| can_view={channel.permissions_for(guild.me).view_channel} "
                f"| can_read_history={channel.permissions_for(guild.me).read_message_history}"
            )

    print("\nWaiting for ANY message...")


@client.event
async def on_message(message):

    if message.author == client.user:
        return

    print("\n" + "=" * 60)
    print("MESSAGE RECEIVED")
    print("=" * 60)

    print("Server:", message.guild.name if message.guild else "DM")
    print("Channel:", message.channel.name)
    print("Channel ID:", message.channel.id)
    print("Author:", message.author)
    print("Content:", repr(message.content))

    if message.channel.id == EXPECTED_CHANNEL_ID:
        print("✅ THIS IS THE CONFIGURED CHANNEL")
    else:
        print("⚠️ MESSAGE CAME FROM A DIFFERENT CHANNEL")


client.run(TOKEN)
