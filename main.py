from src.data.preprocess import preprocess_gpu_data
from src.models.train_model import train
from src.data.load_data import load_model
from src.inference.router import answer_fps_query
from src.llm.handler import handle_natural_language_query

model = load_model()


def main():
    print("GPU Advisor (type 'exit' to quit)\n")

    while True:
        message = input("Ask: ").strip()
        if not message:
            continue

        if message.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        if not message:
            continue

        try:

            result = handle_natural_language_query(message, model)

            print("\nStructured input:")
            print(result["structured_input"])

            print("\nBackend result:")
            print(result["backend_result"])

            print("\nFinal response:")
            print(result["response"])
            print()
        except Exception as e:
            print(f'\nError {e}\n')


if __name__ == "__main__":
    main()
