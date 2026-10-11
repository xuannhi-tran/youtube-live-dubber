import os
import wave
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

text = (
    "Theo bạn, để có được hạnh phúc, "
    "tiền quan trọng hơn hay tình yêu quan trọng hơn?"
)

voices = [
    "Charon",
    "Fenrir",
    "Orus",
    "Puck",
]


def save_wav(filename, audio_data):
    with wave.open(filename, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(audio_data)


for voice in voices:
    print(f"Generating {voice}...")

    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash-tts",
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice
                        )
                    )
                ),
            ),
        )

        audio_data = (
            response.candidates[0]
            .content.parts[0]
            .inline_data.data
        )

        filename = f"voice_{voice.lower()}.wav"

        save_wav(filename, audio_data)

        print(f"✓ Saved {filename}")

    except Exception as e:
        print(f"✗ {voice} failed:")
        print(e)


print("\nDone.")