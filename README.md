# Robust Agentic RAG v2.0 🤖

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-1.2.9-green)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0.8-orange)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Memory-blue)](https://www.postgresql.org/)

A robust **Agentic RAG** system built with **LangGraph**. Unlike traditional linear RAG chains, this system utilizes a graph-based orchestration with **PostgreSQL-backed long-term memory** to provide adaptive capabilities such as **Intent Routing**, **Corrective RAG (CRAG)**, **Web Search Fallback**, **Hallucination Detection**, and **Utility Checks**. It aims to solve common RAG issues like retrieval inaccuracy, hallucinations, and context insufficiency.

## ✨ Key Features
* **🧠 Intelligent Intent Routing:** Analyzes user intent before retrieval to decide if RAG is needed and performs **Query Expansion** to improve recall.
* **💾 Persistent Long-Term Memory:** Integrated **PostgreSQL** to store conversation history and user preferences, enabling cross-session context awareness and personalized interactions.
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
    %% --- 样式定义 ---
    classDef db fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef ext fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    classDef decision fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,stroke-dasharray: 5 5;
    classDef term fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;

    %% --- 外部实体与数据库 ---
    User(["👤 User / Client"])
    Postgres["🗄️ PostgreSQL\nLong-term Memory"]:::db
    Chroma["🗄️ ChromaDB\nVector Store"]:::db
    %% 修复：移除了多余的右括号，并添加了双引号
    WebSearch[["🌐 Web Search\n(Bright Data / MCP)"]]:::ext

    %% --- 流程开始 ---
    User --> |"Query"| Start([Start])
    Postgres -.-> |"Load Chat History"| Start

    %% --- Stage 1: 意图与路由 ---
    subgraph Stage1 [Stage 1: Pre-Retrieval]
        Start --> Intent{"Intent Routing"}:::decision
        Intent -- "Chat / Self-Ref" --> NoRetrieval["Pass Through"]:::process
        Intent -- "External Knowledge" --> Rewriter["Query Expansion\n(Generate 2 Queries)"]:::process
    end

    %% --- Stage 2: 检索与评估 ---
    subgraph Stage2 [Stage 2: Retrieval & Evaluation]
        Rewriter --> MultiRetrieve["Multi-Query Retrieval"]:::process
        MultiRetrieve <--> |"Fetch Top-k Docs"| Chroma
        
        MultiRetrieve --> CRAG{"CRAG Evaluator\n(Reranker Model)"}:::decision
        CRAG -- "Correct (>0.7)" --> FilterDocs["Keep High Quality Docs"]:::process
        CRAG -- "Ambiguous (0.3-0.7)" --> WebTrigger["Keep Partial + Trigger Search"]:::process
        CRAG -- "Incorrect (<0.3)" --> DropDocs["Drop Docs + Trigger Search"]:::process
    end

    %% --- Stage 3: 网络搜索 ---
    subgraph Stage3 [Stage 3: Web Search Augmentation]
        WebTrigger --> CallMCP["Call MCP Client"]:::process
        DropDocs --> CallMCP
        CallMCP <--> |"Search Query"| WebSearch
        CallMCP --> Refine["Knowledge Refinement\n(LLM Summarization)"]:::process
    end

    %% --- Stage 4: 生成 ---
    subgraph Stage4 [Stage 4: Generation]
        NoRetrieval --> Generator["LLM Generator"]:::process
        FilterDocs --> Generator
        Refine --> Generator
        
        Postgres -.-> |"Inject History Context"| Generator
    end

    %% --- Stage 5: 反思与闭环 ---
    subgraph Stage5 [Stage 5: Reflection Loop]
        Generator --> SupportCheck{"Hallucination Check\n(Reranker)"}:::decision
        
        SupportCheck -- "Unsupported (<0.3)" --> RetryGen["Retry Generation\n(Max 2 times)"]:::process
        RetryGen --> Generator
        
        SupportCheck -- "Supported" --> UtilityCheck{"Utility Check\n(LLM Judge)"}:::decision
        
        UtilityCheck -- "Not Useful" --> RewriteLoop["Rewrite Query (v2)"]:::process
        RewriteLoop --> CheckSim{"Similarity Check"}:::decision
        
        CheckSim -- "High Sim (Loop)" --> ForceWeb["Force Web Search"]:::process
        CheckSim -- "New Angle" --> Backtrack["Backtrack to Stage 2"]:::process
        
        Backtrack --> MultiRetrieve
        ForceWeb --> CallMCP
    end

    %% --- 结束 ---
    UtilityCheck -- "Useful" --> End([End / Final Answer]):::term
    
    %% --- 存储记忆 ---
    End -.-> |"Save Turn"| Postgres
