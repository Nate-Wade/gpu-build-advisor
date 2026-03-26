import streamlit as st
from main import run_gpu_advisor

st.set_page_config(page_title="GPU Advisor", page_icon="🎮", layout="centered")

st.title("GPU Advisor")
st.caption("Ask questions about GPU selection and FPS performance for specific games, resolutions, and settings.")
st.caption(
    "***FPS estimates are based on benchmark data and machine learning predictions. GPU prices reflect original launch prices and may differ from current market values."
)
if "history" not in st.session_state:
    st.session_state.history = []


def format_settings_name(name):
    if not name:
        return "N/A"

    name = name.replace(" settings", "")

    return name


def render_metrics(result):
    backend = result.get("backend_result", {})
    structured = result.get("structured_input", {})

    fps_value = backend.get("fps") or backend.get("predicted_fps")
    gpu_value = backend.get("gpu_name") or structured.get("gpu_name", "N/A")
    resolution_value = structured.get("Resolution", "N/A")
    setting_value = structured.get("Setting", "N/A")
    price_value = backend.get("max_price")

    gpu_value_raw = backend.get(
        "gpu_name") or structured.get("gpu_name", "N/A")
    setting_clean = format_settings_name(setting_value)

    metric_cols = st.columns(1)

    with metric_cols[0]:
        st.metric("GPU", gpu_value_raw)

    metric_cols = st.columns(4)

    with metric_cols[0]:
        st.metric("FPS", fps_value if fps_value is not None else "N/A")

    with metric_cols[1]:
        st.metric("Resolution", resolution_value)

    with metric_cols[2]:
        st.metric("Setting", setting_clean.capitalize()
                  if isinstance(setting_clean, str) else setting_clean)

    with metric_cols[3]:
        if price_value is not None:
            st.metric("Price", f"${int(price_value)}")
        else:
            st.metric("Price", "N/A")


def render_history():
    for item in st.session_state.history:
        question = item["question"]
        result = item["result"]

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            st.write(result.get("final_response", "No response generated."))
            render_metrics(result)

            # with st.expander("Show parsed input"):
            #     st.json(result.get("structured_input", {}))

            # with st.expander("Show backend result"):
            #     st.json(result.get("backend_result", {}))


top_col1, top_col2 = st.columns([1, 1])

with top_col1:
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()

if "show_example" not in st.session_state:
    st.session_state.show_example = False

with top_col2:
    if st.button("Show Example Prompt", use_container_width=True):
        st.session_state.show_example = not st.session_state.show_example

if st.session_state.show_example:
    st.info(
        "How many FPS will an RTX 3080 get in Cyberpunk 2077 at 1440p on high settings?")

render_history()

user_message = st.chat_input("Ask a GPU performance question")

if user_message:
    with st.chat_message("user"):
        st.write(user_message)

    try:
        with st.spinner("Running GPU Advisor..."):
            result = run_gpu_advisor(user_message)

        st.session_state.history.append({
            "question": user_message,
            "result": result
        })

        with st.chat_message("assistant"):
            st.write(result.get("final_response", "No response generated."))
            render_metrics(result)

            # with st.expander("Show parsed input"):
            #     st.json(result.get("structured_input", {}))

            # with st.expander("Show backend result"):
            #     st.json(result.get("backend_result", {}))

    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Error: {e}")
