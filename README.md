# Robust Agentic RAG v2.0 🤖

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-1.2.9-green)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0.8-orange)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)

A robust **Agentic RAG** system built with **LangGraph**. Unlike traditional linear RAG chains, this system utilizes a graph-based orchestration to provide adaptive capabilities such as **Intent Routing**, **Corrective RAG (CRAG)**, **Web Search Fallback**, **Hallucination Detection**, and **Utility Checks**. It aims to solve common RAG issues like retrieval inaccuracy, hallucinations, and context insufficiency.

## ✨ Key Features

* **🧠 Intelligent Intent Routing:** Analyzes user intent before retrieval to decide if RAG is needed and performs **Query Expansion** to improve recall.
* **⚖️ CRAG (Corrective RAG):** Evaluates retrieved documents using a Reranker model.
    * **Correct:** Uses retrieved documents directly.
    * **Ambiguous:** Combines retrieved documents with Web Search results.
    * **Incorrect:** Discards retrieved documents and relies solely on Web Search.
* **🌐 Web Search Enhancement (MCP):** Integrates **Model Context Protocol (MCP)** and **Bright Data** to fetch fresh information when the local knowledge base is insufficient.
* **🛡️ Self-Reflection & Hallucination Check:**
    * **Support Check:** Verifies if the generated answer is supported by the documents to minimize hallucinations.
    * **Utility Check:** Evaluates if the answer actually addresses the user's query. If not, it rewrites the query and retries.
* **📝 Knowledge Refinement:** Extracts core bullet points from verbose documents to reduce context noise.
* **💬 Streaming UI:** A Streamlit-based chat interface that visualizes the Agent's thought process (Graph execution path) in real-time.

## 🏗️ System Architecture (Workflow)

The system implements a state machine using LangGraph:

```mermaid
graph TD
    Start([Start]) --> Routing[Stage 1: Intent Routing & Expansion]
    
    Routing -- No Retrieval --> Generate
    Routing -- Need Retrieval --> Retrieve[Stage 2: Vector Store Retrieval]
    
    Retrieve --> CRAG[Stage 2: CRAG Evaluator]
    
    CRAG -- Correct --> Generate
    CRAG -- Ambiguous/Incorrect --> WebSearch[Stage 3: Web Search]
    
    WebSearch --> Generate[Stage 4: Generation]
    
    Generate --> SupportCheck[Stage 5: Support/Hallucination Check]
    
    SupportCheck -- Hallucination Detected (Retry) --> Generate
    SupportCheck -- Supported --> UtilityCheck[Stage 5: Utility Check]
    
    UtilityCheck -- Useful --> End([End])
    UtilityCheck -- Not Useful (Rewrite Query) --> Retrieve
