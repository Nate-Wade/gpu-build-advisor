import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))


def generate_final_response(user_message, structured_input, backend_result):
    """
    Generate the final natural language response from backend output
    """

    response = client.responses.create(
        model="gpt-5-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "You are a GPU performance assistant. "
                    "Use the backend result value exactly as given"
                    "Do not change numeric values. "
                    "Do not invent GPUs, FPS, prices, or assumptions. "
                    "Explain the result naturally and clearly."
                    # "Don't include observed or source or a note"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original user message: \n{user_message}\n\n"
                    f"Structured input: \n{json.dumps(structured_input, indent=2)}\n\n"
                    f"Backend result: \n{json.dumps(backend_result, indent=2)}\n\n"
                    "Write the final answer to the user."
                ),
            },
        ],
    )

    return response.output_text
