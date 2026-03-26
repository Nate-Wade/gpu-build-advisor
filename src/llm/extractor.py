import json
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=st.secrets["OPEN_AI_API_KEY"])


QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "gpu_name": {"type": ["string", "null"]},
        "Game_Name": {"type": ["string", "null"]},
        "Resolution": {"type": ["string", "null"]},
        "Setting": {"type": ["string", "null"]},
        "fps": {"type": ["number", "null"]},
        "max_price": {"type": ["number", "null"]}
    },
    "required": [
        "gpu_name",
        "Game_Name",
        "Resolution",
        "Setting",
        "fps",
        "max_price"
    ],
    "additionalProperties": False
}


def extract_query_fields(user_message: str) -> dict:
    """
    Extract structured GPU query fields from a natural language message.
    """
    response = client.responses.create(
        model="gpt-5-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "You extract GPU advisor requests into structured JSON. "
                    "Return only values explicitly stated or strongly implied. "
                    "Use null for missing fields. "
                    "Do not add extra keys."
                ),
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "gpu_query",
                "strict": True,
                "schema": QUERY_SCHEMA,
            }
        },
    )

    output_text = response.output_text
    return json.loads(output_text)
