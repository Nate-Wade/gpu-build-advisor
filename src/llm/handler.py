from src.llm.extractor import extract_query_fields
from src.llm.responder import generate_final_response
from src.inference.router import answer_fps_query


def clean_user_input(user_input):
    return {k: v for k, v in user_input.items() if v is not None}


def handle_natural_language_query(user_message: str, model) -> dict:
    """
    End to end LLM workflow:
    1. Extract structured fields
    2. Call backend inference
    3. Generate final response
    """
    structured_input = extract_query_fields(user_message)
    # print(f"Extracted structured input: {structured_input}")
    structured_input = clean_user_input(structured_input)
    # print(f"Extracted cleaned structured input: {structured_input}")
    backend_result = answer_fps_query(structured_input, model)
    final_response = generate_final_response(
        user_message=user_message,
        structured_input=structured_input,
        backend_result=backend_result,
    )

    return {
        "structured_input": structured_input,
        "backend_result": backend_result,
        "response": final_response,
    }
