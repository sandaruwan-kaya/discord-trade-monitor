from llm_parser import parse_discord_message

message = """
BTC is holding the 82,900 support area.
A break above 84,600 should open the door toward 85,900.
Below 81,600 invalidates the bullish setup.
"""

signal = parse_discord_message(message)

print(signal.model_dump_json(indent=2))
