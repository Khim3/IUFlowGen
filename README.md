# IUFlowGen

**AI-Powered Flowchart Generation System**  
_Convert complex procedural documents into structured, interactive flowcharts using local LLMs, retrieval-augmented generation (RAG), and DOT-based graph visualization._

![IUFlowGen Screenshot](./docs/sample_flowchart.png) <!-- Optional: add your image -->

---

## 📘 Abstract

Procedural documents in technical, legal, and compliance domains are often difficult to understand due to their length, logic complexity, and domain-specific language. **IUFlowGen** is a modular, AI-assisted system that automatically transforms such documents into **structured flowcharts**, enabling better comprehension, traceability, and decision-making.

Unlike cloud-dependent solutions, IUFlowGen runs entirely **offline**, ensuring **data privacy** and **token-free usage**. It features **interactive visualization**, **clarification queries**, and **human-in-the-loop validation** to support high-quality procedural understanding.

---

## 🔧 Features

- ✅ End-to-end flowchart generation from complex procedural text
- 🧠 Local LLM integration via Ollama (e.g., `phi3`, `deepseek`)
- 🔍 LightRAG: semantic + structural retrieval for context precision
- 🧰 DOT + Graphviz flowchart generation with beautification
- 🧭 Interactive frontend for review, querying, zooming, and toggling layouts
- 🔐 100% local execution — no API keys or internet required

---

## 📂 Folder Structure
Here's a breakdown of the project's directory structure:

-   `backend/`: The core logic for text processing, RAG implementation, and the algorithms responsible for generating the flowchart graphs.
-   `frontend/`: The web application built with Streamlit, providing the user interface and incorporating D3.js for rendering interactive flowcharts.
-   `models/`: Dedicated directory for storing Ollama models and their respective configuration files used in the backend.
-   `data/`: Contains sample procedural documents that can be used for testing, demonstration, and training purposes.
-   `scripts/`: Various utility scripts for tasks such as data preprocessing, environment setup, and deployment automation.
-   `docs/`: Comprehensive project documentation, including the main thesis PDF, design documents, and illustrative screenshots.
-   `README.md`: This file, providing an overview and guide to the IUFlowGen project.

---