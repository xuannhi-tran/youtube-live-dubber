import os
import json

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

segments = [
    {
        "start": 0.00,
        "end": 4.00,
        "text": "你认为想要幸福,钱更重要,还是爱情更重要?"
    },
    {
        "start": 4.00,
        "end": 7.00,
        "text": "钱"
    },
    {
        "start": 7.00,
        "end": 15.43,
        "text": "1938年,哈佛大学进行了一个研究关于成长的人的发展"
    },
    {
        "start": 15.43,
        "end": 18.43,
        "text": "研究关于幸福的因素"
    },
    {
        "start": 18.43,
        "end": 21.43,
        "text": "追踪了75年来的72个男人"
    },
    {
        "start": 21.43,
        "end": 25.81,
        "text": "研究中还有一个美国总统,肯尼迪"
    },
]

prompt = f"""
You are the translation component of a real-time video dubbing system.

Translate the following spoken Mandarin Chinese segments into natural
spoken Vietnamese.

Important requirements:
- Preserve the meaning of the speaker.
- Translate for natural Vietnamese speech, not word-for-word subtitles.
- Use the surrounding segments as context.
- The Chinese transcript comes from speech recognition and may contain
  minor recognition errors. Correct an error only when the intended
  meaning is strongly supported by context.
- Do not invent facts to repair uncertain transcript text.
- Keep each Vietnamese segment concise enough to be spoken approximately
  within its original start/end time.
- Preserve all start and end timestamps exactly.
- Do not merge or split segments.
- Return ONLY valid JSON.
- Use this exact structure:
[
  {{
    "start": 0.0,
    "end": 0.0,
    "text": "Vietnamese translation"
  }}
]

Chinese segments:
{json.dumps(segments, ensure_ascii=False)}
"""

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt,
)

text = response.text.strip()

# Gemini may occasionally wrap JSON in a Markdown fence.
if text.startswith("```json"):
    text = text[7:]
elif text.startswith("```"):
    text = text[3:]

if text.endswith("```"):
    text = text[:-3]

translated = json.loads(text.strip())

print(json.dumps(translated, ensure_ascii=False, indent=2))