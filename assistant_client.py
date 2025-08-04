import os
import time
import json
from typing import Optional

from dotenv import load_dotenv, set_key, find_dotenv
from openai import OpenAI
from fastapi import FastAPI, HTTPException
import uvicorn

from system_prompt import SystemPrompt

# Load environment (used for persisting ASSISTANT_ID)
env_path = find_dotenv()
if not env_path:
    open(".env", "a").close()
    env_path = ".env"
load_dotenv(env_path)

MODEL = "gpt-4o"

#OpenAI client setup
OPENAI_API_KEY = os.getenv("ZAP_OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OpenAI API key missing in environment; set ZAP_OPENAI_API_KEY or OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)



# Assistant ID persistence
ASSISTANT_ID: Optional[str] = os.getenv("ASSISTANT_ID")

def persist_assistant_id(assistant_id: str, env_path: str = env_path):
    set_key(env_path, "ASSISTANT_ID", assistant_id)
    os.environ["ASSISTANT_ID"] = assistant_id

def get_field(obj, field: str):
    if hasattr(obj, field):
        return getattr(obj, field)
    if isinstance(obj, dict):
        if field in obj:
            return obj[field]
        nested = obj.get("data")
        if isinstance(nested, dict) and field in nested:
            return nested[field]
    try:
        return getattr(obj, "__dict__", {}).get(field)
    except Exception:
        return None

def ensure_assistant() -> str:
    global ASSISTANT_ID
    if ASSISTANT_ID:
        return ASSISTANT_ID

    prompt = SystemPrompt()
    assistant_obj = client.beta.assistants.create(
        name="ProductPersonalizationAssistant",
        instructions=prompt.get_instructions(),
        model=MODEL,
        description="Assistant that generates a personalized product description paragraph."
    )

    assistant_id_value = get_field(assistant_obj, "id")
    if not assistant_id_value:
        raise RuntimeError(f"Cannot extract assistant ID from response: {assistant_obj!r}")

    ASSISTANT_ID = assistant_id_value
    print(f"[info] Created assistant: {ASSISTANT_ID}")
    persist_assistant_id(ASSISTANT_ID)
    return ASSISTANT_ID

def extract_assistant_reply_from_run(run: dict) -> str:
    output = get_field(run, "output") or {}
    messages = output.get("messages") if isinstance(output, dict) else None
    if messages:
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content") or msg.get("text") or ""
                if isinstance(content, dict):
                    return content.get("value", "")
                elif isinstance(content, list):
                    return " ".join(str(c) for c in content)
                else:
                    return str(content)
    # fallback: retrieve last thread message
    thread_id = get_field(run, "thread_id")
    if not thread_id:
        return ""
    try:
        msgs_resp = client.beta.threads.messages.list(thread_id=thread_id, order="asc")
        data = getattr(msgs_resp, "data", None)
        if data:
            last = data[-1]
            content_blocks = getattr(last, "content", []) or []
            if content_blocks:
                first_block = content_blocks[0]
                text = first_block.get("text", {}).get("value", "")
                if text:
                    return text
                return str(first_block)
    except Exception:
        pass
    return ""

def build_user_message(product_info: dict, user_personal_info: dict, user_provided_info: dict) -> str:
    prompt = SystemPrompt()
    context_json = prompt.build_context_message(
        product_info=product_info,
        user_personal_information=user_personal_info,
        user_provided_information=user_provided_info
    )

    trigger = "כתוב תיאור מוצר מותאם אישית בעברית לפי ההוראות וההקשר." #  trigger for the assistant to generate the description
    return f"{context_json}\n\n{trigger}"

# ---- Hard-coded JSON inputs ----
PRODUCT_JSON = {
    "name": "Apple Watch Ultra 2",
    "category": "Smartwatch",
    "model": "49mm Titanium Case Ocean Band",
    "brand": "Apple",
    "release_date": "10/2023",
    "connectivity": {
        "gps": True,
        "cellular": True,
        "bluetooth": True,
        "wi_fi": True,
        "nfc": True
    },
    "operating_system": "watchOS",
    "display": {
        "size_inches": 1.92,
        "resolution": "502x410"
    },
    "camera": False,
    "mp3_player": False,
    "storage": {
        "internal": "32GB",
        "expandable": False
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

USER_PERSONAL_JSON = {
    "first_name": "Andrey",
    "last_name": "Khoroshkeev",
    "gender": "male",
    "age_range": "25-30",
    "location": "Tel Aviv, Israel",
    "marital_status": "single",
    "children": False,
    "household_income": "average",
    "job_title": "Marketing Director",
    "industry": "Technology",
    "interests": ["Travel", "Wellness", "Outdoor Activities", "Fashion"],
    "hobbies": ["Gaming", "Fitness", "Smart Home Devices"],
    "purchasing_power": "high",
    "shopping_behavior": "early adopter",
    "preferred_devices": ["iPhone", "MacBook Pro"]
}

USER_PROVIDED_JSON = {
    "main_use": "Fitness",
    "primary_activity": "Running",
    "performance_level": "High accuracy"
}

# FastAPI setup
app = FastAPI()

@app.get("/describe-product")
def describe_product():
    assistant_id = ensure_assistant()

    # Create a fresh thread
    try:
        thread = client.beta.threads.create()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Thread creation failed: {e}")
    thread_id = get_field(thread, "id")
    if not thread_id:
        raise HTTPException(status_code=500, detail="Failed to extract thread ID")

    # Build user message (context + trigger)
    full_user_message = build_user_message(PRODUCT_JSON, USER_PERSONAL_JSON, USER_PROVIDED_JSON)

    # Send user message
    try:
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=full_user_message
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send user message: {e}")

    # Start assistant run
    try:
        run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assistant run failed to start: {e}")

    # Poll until completion with extended timeout and diagnostics
    start = time.time()
    timeout_seconds = 30.0
    run_status = get_field(run, "status")
    last_status = run_status
    while run_status not in ("succeeded", "failed"):
        if time.time() - start > timeout_seconds:
            raise HTTPException(
                status_code=504,
                detail={
                    "error": "Assistant run timed out",
                    "last_status": last_status,
                    "run_snapshot": str(run),
                    "run_output": get_field(run, "output"),
                }
            )
        time.sleep(0.5)
        try:
            run_id = get_field(run, "id")
            if not run_id:
                raise RuntimeError("Missing run ID during polling")
            run = client.beta.threads.runs.get(thread_id=thread_id, run_id=run_id)
            run_status = get_field(run, "status")
            if run_status != last_status:
                print(f"[debug] run status changed: {last_status} -> {run_status}")
                last_status = run_status
        except Exception as e:
            print(f"[warn] error fetching run status: {e}")
            time.sleep(0.2)
            run_status = get_field(run, "status") or last_status

    if run_status == "failed":
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Assistant run failed",
                "run": str(run),
                "output": get_field(run, "output"),
            }
        )

    assistant_reply = extract_assistant_reply_from_run(run)
    if not assistant_reply:
        raise HTTPException(status_code=502, detail={
            "error": "No reply extracted from assistant",
            "run_output": get_field(run, "output"),
        })

    return {
        "assistant_reply": assistant_reply,
        "thread_id": thread_id
    }

# Run with Uvicorn
if __name__ == "__main__":
     uvicorn.run("assistant_client:app", host="127.0.0.1", port=8000, reload=True)
