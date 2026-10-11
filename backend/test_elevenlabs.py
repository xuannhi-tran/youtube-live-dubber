import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()

client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY")
)

VOICE_ID = "ZsjEJaLQy3sgvwxicmDx"

text = (
    "Theo bạn, để có được hạnh phúc, "
    "tiền quan trọng hơn hay tình yêu quan trọng hơn?"
)

print("Generating Vietnamese speech...")

audio = client.text_to_speech.convert(
    voice_id=VOICE_ID,
    text=text,
    model_id="eleven_flash_v2_5",
    output_format="mp3_44100_128",
)

with open("test_elevenlabs.mp3", "wb") as f:
    for chunk in audio:
        f.write(chunk)

print("Saved test_elevenlabs.mp3")