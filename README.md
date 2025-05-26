# IUFlowGen

**IUFlowGen** is a thesis project designed to convert complex procedural documents into structured flowcharts using AI. It leverages large language models, semantic retrieval, and flowchart rendering to help users understand procedural logic visually and interactively.

## Features

- 🧠 AI-Powered Text Understanding with Qwen + LightRAG
- 🔍 Semantic Retrieval using FAISS and Nomic Embeddings
- 🕸️ Graph Construction with NetworkX
- 🖼️ Flowchart Rendering with Graphviz (DOT)
- 🧑‍💻 Human-in-the-loop Feedback Mechanism

## System Architecture
[Input Document]
↓
[Embedding + Chunking]
↓
[Retrieval-Augmented Generation (LightRAG)]
↓
[Graph Assembly (Steps, Actors, Transitions)]
↓
[DOT Code Generation]
↓
[Rendered Flowchart]

## Installation

```bash
git clone https://github.com/Khim3/IUFlowGen.git
cd IUFlowGen
pip install -r requirements.txt
```

## Usage
streamlit run app.py

## File Structure


