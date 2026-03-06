# GPU Build Advisor

Conversational GPU build advisor that answers hardware questions and predicts gaming performance using benchmark data and an XGBoost model.

---

## Overview

GPU Build Advisor is an interactive system that helps users evaluate GPU hardware and gaming performance. The project combines structured benchmark data with a machine learning model to answer questions about GPU builds, compare hardware, and estimate gaming FPS.

The system provides data-driven responses to hardware questions through a conversational interface.

Example questions include:

- What GPU is best for 1440p gaming under $500?
- How much FPS improvement will I get upgrading from a 3060 to a 4070?
- Can this build run Cyberpunk 2077 at 1440p high settings?
- Compare the RTX 4070 and RX 7800 XT.

---

## System Architecture

The system consists of three main components.

### Data Layer

Structured GPU benchmark datasets containing:

- GPU specifications
- Performance benchmarks
- Price information
- Hardware characteristics

### Machine Learning Model

An **XGBoost regression model** trained to estimate gaming FPS based on GPU specifications and system configuration.

### Chat Interface

A conversational interface that interprets user questions and routes them to backend functions such as:

- GPU comparison
- Build recommendation
- Performance prediction

---

## Example Workflow

1. User asks a question about GPU hardware or gaming performance.
2. The system extracts relevant parameters from the query.
3. The backend retrieves hardware data or runs the ML prediction model.
4. The system returns a natural language explanation of the results.
