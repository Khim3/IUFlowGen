# IUFlowGen

**AI-Powered Flowchart Generation System**  
_Convert complex procedural documents into structured, interactive flowcharts using local LLMs, retrieval-augmented generation (RAG), and DOT-based graph visualization._

---

## 📘 Abstract

Procedural documents in technical, legal, and compliance domains are often difficult to understand due to their length, logic complexity, and domain-specific language. **IUFlowGen** is a modular, AI-assisted system that automatically transforms such documents into **structured flowcharts**, enabling better comprehension, traceability, and decision-making.

Unlike cloud-dependent solutions, IUFlowGen runs entirely **offline**, ensuring **data privacy** and **token-free usage**. It features **interactive visualization**, **clarification queries**, and **human-in-the-loop validation** to support high-quality procedural understanding.

---

## 🔧 Features

- ✅ End-to-end flowchart generation from complex procedural text
- 🧠 Local LLM integration via Ollama (e.g., `phi4`, `deepseek-r1`)
- 🔍 LightRAG: semantic + structural retrieval for context precision
- 🧰 DOT + Graphviz flowchart generation with beautification
- 🧭 Interactive frontend for review, querying, zooming, and toggling layouts
- 🔐 100% local execution — no API keys or internet required

---

## 📂 Folder Structure
Here's a breakdown of the project's directory structure:

-   `backend.py`: The core logic for text processing, RAG implementation, and the algorithms responsible for generating the flowchart graphs.
-   `app.py`: The web application built with Streamlit, providing the user interface and incorporating D3.js for rendering interactive flowcharts.
-   `input/`: Contains sample procedural documents that can be used for testing, demonstration, and training purposes.
-   `Slide+Report/`: Comprehensive project documentation, including the main thesis PDF, design documents, and illustrative screenshots.
-    `server/`: Contains the Streamlit server configuration and any additional server-side logic needed for the application.
-   `README.md`: This file, providing an overview and guide to the IUFlowGen project.


---
## 🚀 Getting Started

### Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Graphviz installed (`dot` CLI should be available)
- `pip install -r requirements.txt`
### Setup

```bash
# Clone repository
git clone https://github.com/Khim3/IUFlowGen.git
cd IUFlowGen

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

## 🧠 System Pipeline

1. 📝 **User uploads a procedural document**  
   The system accepts machine-readable PDFs with procedural content.

2. ⚙️ **Text is chunked and embedded**  
   The document is split into semantically meaningful chunks and encoded using vector embeddings for semantic similarity and graph structure.

3. 🧠 **LLM is prompted with LightRAG-augmented input**  
   The large language model (e.g., Phi-4 or DeepSeek via Ollama) receives contextually retrieved snippets via LightRAG to generate relevant procedural steps.

4. 🧾 **Regex filtering and DOT code post-processing**  
   The raw LLM output is normalized using regular expressions to extract valid node-action-entity relationships in DOT syntax.

5. 📊 **Graphviz renders the flowchart**  
   The DOT code is visualized into an interactive flowchart using Graphviz, displayed with D3.js in the frontend.

6. 🙋‍♂️ **User explores and verifies**  
   Users interact with the diagram (zoom, pan, query) and validate the AI-generated structure in a human-in-the-loop review process.

---

## 📊 Evaluation

IUFlowGen was evaluated through a structured user study involving procedural documents of varying complexity:

- 🧪 **Level 1–4 Documents**: Ranging from synthetic to real-world procedural text.
- 👥 **Control vs Assisted Groups**: One group manually created flowcharts, while the other used IUFlowGen.
- 📝 **Evaluation Metrics**:
  - **Accuracy**: Correctness of procedural structure
  - **Completeness**: Coverage of all steps and relationships
  - **Efficiency**: Time taken to produce or refine a flowchart

**Result**:  
IUFlowGen significantly improved user performance—reducing effort, increasing structural precision, and aiding clarity—especially with ambiguous or complex source documents.

## 🤝 Acknowledgments

- **Dr. Tran Thanh Tung** — Thesis Supervisor whose mentorship and insight shaped the foundation of this project.
- **Central Interdisciplinary Laboratory** — For providing infrastructure and computational resources at International University – VNU-HCM.
- **Open-source Tools and Frameworks** —  
  IUFlowGen was made possible thanks to the contributions of the open-source community, particularly:
  - [Graphviz](https://graphviz.org) — For DOT-based flowchart rendering.
  - [Ollama](https://ollama.com) — For running local LLMs like Phi and DeepSeek.
  - [LangChain](https://www.langchain.com) — For prompt orchestration and retrieval chaining.

