# GPU Performance Advisor

An interactive machine learning application that predicts GPU performance (FPS) across games, resolutions, and graphics settings.

## Features
- Predict FPS using an XGBoost model trained on GPU benchmark data
- Query using natural language (e.g., "RTX 3080 Cyberpunk 1440p high")
- Hybrid inference: observed benchmarks + model predictions
- Reverse lookup: find GPUs for a target FPS or budget
- Interactive Streamlit UI

## Tech Stack
- Python, Pandas, Scikit-learn, XGBoost
- Streamlit (frontend)
- OpenAI API (natural language parsing)
- Custom backend routing + fuzzy matching

## Example Queries
- RTX 3080 Cyberpunk 1440p high
- Best GPU under $500
- GPU for 120 FPS in Fortnite

## How it works
1. User query → structured extraction (LLM)
2. GPU + parameters resolved via fuzzy matching
3. Backend routes:
   - Observed benchmark lookup
   - ML prediction
   - Reverse GPU search
4. Results displayed in UI