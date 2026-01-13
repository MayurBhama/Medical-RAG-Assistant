# Medical RAG Assistant

A sophisticated **Medical Research Assistant** powered by **RAG (Retrieval-Augmented Generation)** that answers medical questions using real research papers from PubMed. Built with LangChain, Ollama, and ChromaDB.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [How It Works](#how-it-works)
- [Agent System](#agent-system)
- [RAG Pipeline](#rag-pipeline)
- [Fallback Mechanism](#fallback-mechanism)
- [Guardrails](#guardrails)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Evaluation (RAGAS)](#evaluation-ragas)
- [Configuration](#configuration)
- [Future Improvements](#future-improvements)

---

## Overview

The **Medical RAG Assistant** is an AI-powered research assistant that:

1. **Retrieves** relevant medical research from a vector database (ChromaDB)
2. **Augments** the LLM's knowledge with real PubMed papers
3. **Generates** accurate, grounded answers to medical questions

Unlike generic chatbots that hallucinate, this system **only answers from verified medical literature**, making it suitable for research and educational purposes.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **PubMed Integration** | Fetches real research papers from NCBI PubMed database |
| **RAG Architecture** | Combines retrieval with generation for factual answers |
| **ReAct Agent** | Intelligent agent that decides when to search vs. retrieve |
| **Medical Guardrails** | Refuses non-medical questions and creative requests |
| **RAGAS Evaluation** | Quantitative evaluation of answer quality |
| **Smart Fallback** | Automatically searches PubMed when local knowledge is insufficient |
| **Streamlit UI** | Clean, user-friendly chat interface |

---

## Architecture

```
+-------------------------------------------------------------------------+
|                           USER INTERFACE                                 |
|                         (Streamlit - app.py)                            |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                           AGENT LAYER                                    |
|                    (LangChain ReAct Agent - agent.py)                   |
|  +------------------------------------------------------------------+   |
|  |  Decision: Which tool to use?                                    |   |
|  |  - query_medical_knowledge -> Search existing database           |   |
|  |  - search_pubmed -> Fetch new research from PubMed               |   |
|  |  - ingest_new_research -> Add new papers to database             |   |
|  +------------------------------------------------------------------+   |
+-------------------------------------------------------------------------+
                                    |
                    +---------------+---------------+
                    v               v               v
         +--------------+  +--------------+  +--------------+
         |   RAG CHAIN  |  |  PUBMED API  |  |  INGESTION   |
         |   (core.py)  |  |  (utils.py)  |  | (ingest.py)  |
         +--------------+  +--------------+  +--------------+
                 |                                    |
                 v                                    v
         +--------------+                    +--------------+
         |   ChromaDB   |<-------------------|  Embeddings  |
         | (Vector Store)|                   | (HuggingFace)|
         +--------------+                    +--------------+
                 |
                 v
         +--------------+
         |  Ollama LLM  |
         | (Llama 3.1)  |
         +--------------+
                 |
                 v
         +--------------+
         |   RESPONSE   |
         +--------------+
```

---

## Technology Stack

### Why Each Technology Was Chosen

| Technology | Purpose | Why This Choice |
|------------|---------|-----------------|
| **LangChain** | Orchestration framework | Industry standard for RAG apps, excellent abstractions for chains and agents |
| **Ollama + Llama 3.1:8b** | Local LLM | Free, private, no API costs, good medical understanding |
| **ChromaDB** | Vector database | Lightweight, embedded, perfect for local development |
| **HuggingFace Embeddings** | Text embeddings | `all-MiniLM-L6-v2` is fast and effective for semantic search |
| **Biopython (Entrez)** | PubMed API | Official library for NCBI databases, reliable |
| **Streamlit** | Web UI | Rapid prototyping, built-in chat components |
| **RAGAS** | Evaluation | Standard framework for RAG evaluation metrics |

### Dependencies

```
langchain
langchain-ollama
langchain-huggingface
langchain-community
langchain-text-splitters
langchain-classic
langchain-core
chromadb
sentence-transformers
biopython
tf-keras
streamlit
python-dotenv
ragas
datasets
nest-asyncio
rapidfuzz
```

---

## How It Works

### Step-by-Step Flow

```
1. USER ASKS QUESTION
   "What are the side effects of immunotherapy for lung cancer?"

2. AGENT RECEIVES QUESTION
   - Analyzes the question and decides on action
   - Thought: "I should query the medical knowledge base"

3. TOOL SELECTION
   - Agent selects: query_medical_knowledge

4. RAG PIPELINE EXECUTES
   a) EMBEDDING: Question -> 384-dim vector
   b) RETRIEVAL: Find top-5 similar chunks from ChromaDB
   c) CONTEXT ASSEMBLY: Combine retrieved chunks
   d) GENERATION: LLM generates answer from context

5. RESPONSE VALIDATION
   - Check if answer is grounded in context
   - Apply guardrails (no hallucination, no off-topic)

6. FALLBACK (if needed)
   - If no local answer -> Search PubMed automatically
   - Summarize new research for user

7. RETURN ANSWER TO USER
```

---

## Agent System

### ReAct Agent Architecture

The agent uses the **ReAct (Reasoning + Acting)** paradigm:

```
Question: What is CAR-T therapy?
Thought: I need to query the medical knowledge base for CAR-T therapy information.
Action: query_medical_knowledge
Action Input: "What is CAR-T therapy?"
Observation: CAR-T cell therapy is a type of immunotherapy...
Thought: I now have the answer from the knowledge base.
Final Answer: CAR-T cell therapy is a type of immunotherapy...
```

### Available Tools

| Tool | Function | When Used |
|------|----------|-----------|
| `query_medical_knowledge` | Queries ChromaDB vector store | User asks about topics in database |
| `search_pubmed` | Fetches papers from PubMed API | User asks about new/unknown topics |
| `ingest_new_research` | Adds papers to ChromaDB | User wants to expand knowledge base |

### Agent Code (agent.py)

```python
# ReAct prompt with strict guardrails
template = """You are a focused medical research assistant.
Your goal is to answer medical questions using the provided tools.

CRITICAL INSTRUCTION:
If you do not find the answer in your knowledge base, 
you MUST use 'search_pubmed' to find new research.
Do NOT give up without searching first.

You must NOT answer general knowledge questions or creative writing requests.
If a question is off-topic: "I am designed to answer medical research questions only."
...
"""
```

---

## RAG Pipeline

### What is RAG?

**RAG (Retrieval-Augmented Generation)** combines:
- **Retrieval**: Finding relevant documents from a database
- **Generation**: Using an LLM to synthesize an answer

This prevents hallucination by grounding answers in real data.

### RAG Components

```
+-------------------------------------------------------------+
|                      RAG PIPELINE                           |
+-------------------------------------------------------------+
|  1. DOCUMENT LOADING                                        |
|     - PubMed abstracts fetched via Biopython               |
|                                                             |
|  2. TEXT SPLITTING                                          |
|     - RecursiveCharacterTextSplitter                       |
|     - chunk_size: 1000, overlap: 200                       |
|                                                             |
|  3. EMBEDDING                                               |
|     - HuggingFace: all-MiniLM-L6-v2                        |
|     - 384-dimensional vectors                              |
|                                                             |
|  4. VECTOR STORAGE                                          |
|     - ChromaDB (persistent, local)                         |
|                                                             |
|  5. RETRIEVAL                                               |
|     - Similarity search (k=5 chunks)                       |
|     - Cosine similarity matching                           |
|                                                             |
|  6. GENERATION                                              |
|     - Ollama Llama 3.1:8b                                  |
|     - Strict prompt: answer ONLY from context              |
+-------------------------------------------------------------+
```

### Why These Parameters?

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `chunk_size` | 1000 | Large enough for context, small enough for precision |
| `chunk_overlap` | 200 | Prevents information loss at chunk boundaries |
| `k` (retrieved chunks) | 5 | Balance between context richness and noise |
| `temperature` | 0 | Deterministic, factual responses (no creativity) |

---

## Fallback Mechanism

### Three-Tier Fallback System

When the primary method fails, the system has intelligent fallbacks:

```
+-------------------------------------------------------------+
|                    FALLBACK HIERARCHY                        |
+-------------------------------------------------------------+
|                                                              |
|  TIER 1: AGENT                                              |
|  - Agent tries to use tools to answer                       |
|  - May fail due to: iteration limit, parsing errors         |
|                     |                                        |
|                     v (if fails)                             |
|  TIER 2: DIRECT RAG                                         |
|  - Bypass agent, query ChromaDB directly                    |
|  - May fail due to: topic not in database                   |
|                     |                                        |
|                     v (if fails)                             |
|  TIER 3: PUBMED SEARCH                                      |
|  - Search PubMed API for fresh papers                       |
|  - Use LLM to summarize results                             |
|  - Return summarized answer to user                         |
|                                                              |
+-------------------------------------------------------------+
```

### Fallback Code (agent.py)

```python
def run_agent(query: str):
    # TIER 1: Try agent
    response = agent.invoke({"input": query})
    output = response.get("output", "")
    
    # TIER 2: If agent hit iteration limit, try direct RAG
    if "iteration limit" in output.lower():
        answer, docs = get_medical_answer(query)
        if "cannot find" not in answer.lower():
            return answer
        output = answer  # Pass to next tier
    
    # TIER 3: If RAG failed, search PubMed
    if "cannot find information" in output.lower():
        search_result = search_pubmed.invoke(query)
        llm = ChatOllama(model="llama3.1:8b")
        summary = llm.invoke(f"Summarize: {search_result}")
        return summary.content
    
    return output
```

---

## Guardrails

### Strict Medical Focus

The system includes multiple layers of protection:

| Guardrail | Implementation | Purpose |
|-----------|----------------|---------|
| **Context-Only Answers** | Prompt engineering | Prevents hallucination |
| **Off-Topic Rejection** | Agent prompt | Blocks non-medical questions |
| **No Creative Writing** | Agent prompt | Blocks poems, stories, etc. |
| **Grounded Responses** | RAG architecture | Ensures factual accuracy |

### Prompt Guardrails (core.py)

```python
template = """You are a specialized medical research assistant. 
Your expertise is strictly limited to the provided research context.

STRICT GUARDRAILS:
1. Answer ONLY using the information from the Context below.
2. If the Context does not contain the answer, you MUST say: 
   "I cannot find information about this in the current medical database."
3. Do NOT use your internal training data for general knowledge questions.
4. Do NOT perform creative writing tasks (poems, stories, jokes).
5. If the question is about medical advice not found in context, decline.

Context: {context}
Question: {question}
Answer (based strictly on context):
"""
```

---

## Installation

### Prerequisites

- Python 3.10+
- Ollama installed with `llama3.1:8b` model

### Setup Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Medical-RAG-Assistant.git
cd Medical-RAG-Assistant

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Ensure Ollama is running with Llama 3.1
ollama pull llama3.1:8b
ollama serve

# 5. Run the application
streamlit run app.py
```

---

## Usage

### Web Interface (Recommended)

1. Start the app: `streamlit run app.py`
2. Open browser: `http://localhost:8501`
3. Ask medical questions in the chat
4. Add new research topics via sidebar

### Ingest New Research

```python
# Via UI: Use the sidebar "Add New Research"

# Via command line:
from ingest import build_vector_db
build_vector_db("diabetes treatment guidelines 2024")
```

### Run Evaluation

```bash
python evaluate.py
```

---

## Project Structure

```
Medical-RAG-Assistant/
|
|-- app.py              # Streamlit web interface
|                       # - Chat UI with message history
|                       # - Sidebar for research ingestion
|
|-- agent.py            # LangChain ReAct Agent
|                       # - Tool orchestration
|                       # - Fallback mechanism
|                       # - Guardrails enforcement
|
|-- core.py             # RAG Pipeline
|                       # - Vector retrieval from ChromaDB
|                       # - LLM generation with strict prompt
|                       # - Context-grounded answers
|
|-- tools.py            # LangChain Tool Definitions
|                       # - search_pubmed: PubMed API search
|                       # - query_medical_knowledge: ChromaDB query
|                       # - ingest_new_research: Add new papers
|
|-- utils.py            # Utility Functions
|                       # - fetch_pubmed_abstracts: Biopython Entrez
|
|-- ingest.py           # Document Ingestion
|                       # - Fetch from PubMed
|                       # - Split into chunks
|                       # - Embed and store in ChromaDB
|
|-- evaluate.py         # RAGAS Evaluation
|                       # - Automatic question generation
|                       # - Faithfulness, relevancy metrics
|
|-- requirements.txt    # Python dependencies
|-- PROJECT_FLOW.txt    # Detailed architecture documentation
|-- .gitignore          # Git ignore rules
|-- LICENSE             # MIT License
|-- README.md           # This file
```

---

## Evaluation (RAGAS)

### What is RAGAS?

**RAGAS (Retrieval-Augmented Generation Assessment)** is a framework for evaluating RAG systems using LLM-based metrics.

### Metrics Used

| Metric | What It Measures | Our Score |
|--------|------------------|-----------|
| **Faithfulness** | Is the answer grounded in the context? | ~0.80 |
| **Answer Relevancy** | Is the answer relevant to the question? | ~0.86 |
| **Context Precision** | Are retrieved chunks relevant? | ~0.33 |
| **Context Recall** | Does context cover the ground truth? | Varies |

### How Evaluation Works

```python
# 1. Load documents from ChromaDB
documents = get_documents_from_db()

# 2. Generate test questions using LLM
questions, ground_truths = generate_questions_with_llm(documents)

# 3. Run RAG pipeline on each question
for question in questions:
    answer, contexts = get_medical_answer(question)

# 4. Calculate RAGAS metrics
results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
```

### Running Evaluation

```bash
python evaluate.py
```

---

## Configuration

### Key Parameters

| File | Parameter | Value | Purpose |
|------|-----------|-------|---------|
| `ingest.py` | `chunk_size` | 1000 | Size of document chunks |
| `ingest.py` | `chunk_overlap` | 200 | Overlap between chunks |
| `ingest.py` | `max_results` | 25 | Papers fetched per topic |
| `core.py` | `k` | 5 | Retrieved chunks per query |
| `agent.py` | `max_iterations` | 15 | Agent reasoning steps limit |
| `agent.py` | `temperature` | 0 | LLM temperature (deterministic) |

### Changing the LLM

To use a different Ollama model, update these files:

```python
# core.py
llm = ChatOllama(model="your-model-name", temperature=0)

# agent.py
llm = ChatOllama(model="your-model-name", temperature=0)
```

---

## Future Improvements

| Feature | Description | Priority |
|---------|-------------|----------|
| **Cloud Deployment** | Switch to Groq API for free cloud hosting | Medium |
| **Multi-Source RAG** | Add clinical trials, guidelines databases | High |
| **Hybrid Search** | Combine keyword + semantic search | Medium |
| **Reranking** | Add cross-encoder reranking for better precision | Medium |
| **Voice Interface** | Speech-to-text input | Low |
| **Mobile App** | React Native frontend | Low |

---

## License

MIT License - See [LICENSE](LICENSE) file.

---

## Acknowledgments

- LangChain - RAG framework
- Ollama - Local LLM inference
- PubMed/NCBI - Medical literature database
- RAGAS - Evaluation framework

---

Built for Medical Research