import time

from src.data.load_data import load_final_data
from src.llm.extractor import extract_query_fields
from src.llm.responder import generate_final_response
from src.inference.router import answer_fps_query
from src.inference.fuzzy_match import resolve_user_input


def clean_user_input(user_input):
    return {k: v for k, v in user_input.items() if v is not None}


def handle_natural_language_query(user_message: str, model) -> dict:
    """
    End to end LLM workflow:
    1. Extract structured fields
    2. Call backend inference
    3. Generate final response
    """
    t0 = time.time()
    structured_input = extract_query_fields(user_message)
    t1 = time.time()
    # print(f"Extracted structured input: {structured_input}")
    # structured_input = resolve_user_input(structured_input, df)
    structured_input = clean_user_input(structured_input)
    # print(f"Extracted cleaned structured input: {structured_input}")
    backend_result = answer_fps_query(structured_input, model)
    # print(f"Backend result: {backend_result}")
    structured_input["max_price"] = backend_result.get("max_price")
    # print(f"Structured input with max price: {structured_input}")
    t2 = time.time()
    final_response = generate_final_response(
        user_message=user_message,
        structured_input=structured_input,
        backend_result=backend_result,
    )
    t3 = time.time()

    # print("extract:", t1 - t0)
    # print("backend:", t2 - t1)
    # print("final response:", t3 - t2)
    # print("total:", t3 - t0)

    return {
        "structured_input": structured_input,
        "backend_result": backend_result,
        "response": final_response,
    }
