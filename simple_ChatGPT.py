from fastapi.responses import PlainTextResponse
import os
from fastapi import FastAPI, HTTPException
import uvicorn
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware



OPENAI_API_KEY = os.getenv("ZAP_OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OpenAI API key missing; set ZAP_OPENAI_API_KEY or OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = "gpt-4o"

# Combined prompt
PROMPT = """You are a product personalization assistant. Your role is to generate engaging, persuasive, and fully personalized product description paragraphs in Hebrew, aligned to the right for proper Hebrew readability.

You will receive three structured JSON inputs:

Product Information:
Accurately incorporate all product details into the generated paragraph. Do not omit or generalize; include concrete specs, capabilities, and relevant highlights.

User Personal Information:
Use this input sparingly and discreetly.
Do not refer to personal details directly or explicitly. Instead, weave this data into the description in a subtle, emotionally intelligent manner that enhances relevance and resonance—without making the personalization obvious to the user. Avoid any language that could feel invasive, uncanny, or uncomfortable.

User Provided Information:
Use this information prominently and directly.
This input should guide the narrative and form the foundation for relevance and persuasive impact.

You may enrich your output with additional product insights from official, reputable sources only (e.g., verified manufacturer or brand websites). Ensure all external data is factually accurate, current, and complements the provided product details.

Your final output must always:

Be written in fluent Hebrew, aligned right-to-left.

Be a single, cohesive paragraph (not bullet points or lists).

Present the full product offering, integrating all product details accurately.

Create an engaging and persuasive narrative, matching the user's explicitly stated context and needs.

Use a natural, friendly, and enthusiastic tone, as if you are a helpful advisor—not a sales bot.

Respect the user’s privacy by using personal information only in a nuanced way that enhances personalization without drawing attention to it.


User Personal Information 
{
  "first_name": "Andrey",
  "last_name": "Khoroshkeev",
  "gender": "male",
  "age_range": "25-30",
  "location": "Tel Aviv, Israel",
  "marital_status": "single",
  "children": false,
  "household_income": "average",
  "job_title": "Marketing Director",
  "industry": "Technology",
  "interests": [
    "Travel",
    "Wellness",
    "Outdoor Activities",
    "Fashion"
  ],
  "hobbies": [
    "Gaming",
    "Fitness",
    "Smart Home Devices"
  ],
  "purchasing_power": "high",
  "shopping_behavior": "early adopter",
  "preferred_devices": [
    "iPhone",
    "MacBook Pro"
  ]
}

User Provided Information
{
  "main_use": "Fitness",
  "primary_activity": "Running",
  "performance_level": "High accuracy"
}



Product Information:
{
  "name": "Apple Watch Ultra 2",
  "category": "Smartwatch",
  "model": "49mm Titanium Case Ocean Band",
  "brand": "Apple",
  "release_date": "10/2023",
  "connectivity": {
    "gps": true,
    "cellular": true,
    "bluetooth": true,
    "wi_fi": true,
    "nfc": true
  },
  "operating_system": "watchOS",
  "display": {
    "size_inches": 1.92,
    "resolution": "502x410"
  },
  "camera": false,
  "mp3_player": false,
  "storage": {
    "internal": "32GB",
    "expandable": false
  },
  "weight_grams": 61.3,
  "water_dust_resistance": "IP6X",
  "health_features": [
    "ECG",
    "Blood Oxygen (SpO2)",
    "Sleep Tracking",
    "Heart Rate Monitoring"
  ],
  "case_material": "Titanium",
  "band": "Ocean Band"
}
"""

# FastAPI app setup
app = FastAPI()

# CORS middleware to allow Lovable frontend to call the endpoint
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/describe-product", response_class=PlainTextResponse)
def describe_product():
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=1.0,
            max_tokens=700,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Chat completion failed: {e}")

    # Extract the content of the assistant reply
    try:
        choices = getattr(resp, "choices", None) or resp.get("choices", [])
        if not choices:
            raise ValueError("No choices returned from API")

        first = choices[0]
        if isinstance(first, dict):
            message_obj = first.get("message") or {}
        else:
            message_obj = getattr(first, "message", {}) or {}

        if isinstance(message_obj, dict):
            assistant_reply = message_obj.get("content", "")
        elif hasattr(message_obj, "content"):
            assistant_reply = getattr(message_obj, "content") or ""
        else:
            assistant_reply = str(message_obj)

        if not assistant_reply:
            raise ValueError("Empty assistant reply content")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to parse response: {e}")

    return assistant_reply

# Run with Uvicorn
if __name__ == "__main__":
    uvicorn.run("simple_ChatGPT:app", host="127.0.0.1", port=8000, reload=True)
