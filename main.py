from src.data.preprocess import preprocess_gpu_data
from src.models.train_model import train
from src.data.load_data import load_model
from src.inference.router import answer_fps_query
from src.llm.handler import handle_natural_language_query

model = load_model()


def run_gpu_advisor(message):
    if not message or not message.strip():
        raise ValueError("Message cannot be empty.")

    result = handle_natural_language_query(message, model)

    structured_input = result.get("structured_input", {})

    if "Resolution" not in structured_input:
        structured_input["Resolution"] = "1080p"
    if "Setting" not in structured_input:
        structured_input["Setting"] = "medium"

    result["structured_input"] = structured_input

    return {
        "structured_input": result.get("structured_input", {}),
        "backend_result": result.get("backend_result", {}),
        "final_response": result.get("response", "")
    }


def main():
    print("GPU Advisor (type 'exit' to quit)\n")

    while True:
        message = input("Ask: ").strip()

        if not message:
            continue

        if message.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            result = run_gpu_advisor(message)

            print("\nStructured input:")
            print(result["structured_input"])

            print("\nBackend result:")
            print(result["backend_result"])

            print("\nFinal response:")
            print(result["final_response"])
            print()

        except Exception as e:
            print(f"\nError {e}\n")


if __name__ == "__main__":
    main()
