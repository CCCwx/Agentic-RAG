"""
Robust Agentic RAG v2.0 workflow implemented with LangGraph.
Stages: 1 Pre-Retrieval & Routing -> 2 Retrieval & CRAG -> 3 Web Search -> 4 Generation -> 5 Reflection (Support + Utility).
"""
from typing import TypedDict, List, Literal
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from model.factory import chat_model
from rag.vector_store import VectorStoreService
from utils.prompt_loader import (
    load_intent_routing_prompt,
    load_query_expansion_prompt,
    load_generation_prompt,
    load_utility_check_prompt,
)
from utils.reranker import RerankerService
from utils.similarity import query_similarity
from utils.knowledge_refinement import refine_documents
from utils.retrieval_utils import multi_query_retrieve
from agent.agent_tool.web_search import web_search
from schema.schema import IntentResponse, ExpandedQueriesResponse
from utils.logger_handler import logger
import json
import sys

# --- State ---
class RAGState(TypedDict):
    query: str #用户输入的最初始的查询
    query_list: List[str] #初始query + rewrite query
    intent: str #对用于意图的分类
    documents: List[Document] #stage1的从向量数据库检索到的原始文档列表
    refined_context: str # 
    crag_score: float # reranker算出来的分
    crag_decision: Literal["Correct", "Ambiguous", "Incorrect"]
    need_web_search: bool
    web_results: List[Document]
    answer: str
    support_scores: List[float] #记录了每个文档对 answer 的支撑程度
    halluci_generate_counter: int
    overall_generate_counter: int
    rewritten_query_1: str 
    rewritten_query_2: str
    query_sim: float #查询相似度
    utility: str #
    final_answer: str
    next_stage: str #在条件边（Conditional Edge）中使用的字符串


# --- Constants ---
CRAG_CORRECT_THRESHOLD = 0.7
CRAG_AMBIGUOUS_THRESHOLD = 0.3
SUPPORT_THRESHOLD = 0.3
QUERY_SIM_THRESHOLD = 0.9
MAX_HALLUCI_GENERATE = 2
MAX_OVERALL_GENERATE = 2

# 工作流阶段标签（用于实时进度展示）
STAGE_LABELS = {
    "stage1_routing": "阶段1：意图路由 (Pre-Retrieval & Routing)",
    "stage2_retrieve": "阶段2：检索 (Retrieval)",
    "stage2_evaluate_crag": "阶段2：CRAG 评估 (Evaluator)",
    "stage3_web_search": "阶段3：网络搜索 (Web Search)",
    "stage4_generate": "阶段4：生成 (Generation)",
    "stage5_support_check": "阶段5：幻觉检测 (Support Check)",
    "stage5_utility_check": "阶段5：效用评估 (Utility Check)",
}


def _print_stage_debug(node_name: str, update: dict) -> None:
    """将各阶段关键信息打印到终端，便于排查 CRAG/重写/分数 等逻辑。"""
    out = sys.stdout
    if not isinstance(update, dict):
        return
    if node_name == "stage1_routing":
        out.write("\n[终端] === 阶段1 输出 ===\n")
        out.write(f"  intent: {update.get('intent', '')}\n")
        out.write(f"  query_list: {update.get('query_list', [])}\n")
        out.write(f"  rewrite_query_1: {update.get('rewritten_query_1', '')}\n")
        out.flush()
    elif node_name == "stage2_retrieve":
        docs = update.get("documents", [])
        out.write("\n[终端] === 阶段2 检索输出 ===\n")
        out.write(f"  检索文档数: {len(docs)}\n")
        out.flush()
    elif node_name == "stage2_evaluate_crag":
        out.write("\n[终端] === 阶段2 CRAG 评估输出 ===\n")
        out.write(f"  crag_score: {update.get('crag_score', 0)}\n")
        out.write(f"  crag_decision: {update.get('crag_decision', '')}\n")
        out.write(f"  need_web_search: {update.get('need_web_search', False)}\n")
        out.flush()
    elif node_name == "stage5_support_check":
        scores = update.get("support_scores", [])
        out.write("\n[终端] === 阶段5 幻觉检测输出 ===\n")
        out.write(f"  support_scores: {scores}\n")
        out.write(f"  max(support_scores): {max(scores) if scores else 0}\n")
        out.flush()
    elif node_name == "stage5_utility_check":
        out.write("\n[终端] === 阶段5 效用评估输出 ===\n")
        out.write(f"  utility: {update.get('utility', '')}\n")
        if update.get("rewritten_query_2"):
            out.write(f"  rewrite_query_2: {update.get('rewritten_query_2', '')}\n")
        if "query_sim" in update:
            out.write(f"  query_sim (rew1 vs rew2): {update.get('query_sim')}\n")
        out.flush()


# --- Shared services (lazy) ---
_vector_store: VectorStoreService | None = None
_reranker: RerankerService | None = None


def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreService()
    return _vector_store


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = RerankerService()
    return _reranker


# --- Stage 1: Pre-Retrieval & Routing ---
def stage1_routing(state: RAGState) -> RAGState:
    query = state["query"]
    prompt_text = load_intent_routing_prompt()
    prompt = PromptTemplate.from_template(prompt_text)
    chain = prompt | chat_model.with_structured_output(IntentResponse)
    result = chain.invoke({"query": query})
    intent = "No Retrieval" if result and "No Retrieval" in (result.response or "") else "Retrieval Needed"
    if intent == "No Retrieval":
        return {
            **state,
            "intent": intent,
            "query_list": [query],
            "documents": [],
            "refined_context": "",
            "next_stage": "generate",
        }
    # Query expansion: 1-2 rewritten queries
    exp_prompt = load_query_expansion_prompt()
    exp_template = PromptTemplate.from_template(exp_prompt)
    try:
        exp_chain = exp_template | chat_model.with_structured_output(ExpandedQueriesResponse)
        result = exp_chain.invoke({"query": query})
        queries = (result.queries or [query]) if result else [query]
    except Exception:
        queries = [query]
    if not isinstance(queries, list):
        queries = [queries] if queries else [query]
    query_list = [query] + [q for q in queries if q and str(q).strip() and str(q).strip() != query][:1]
    if len(query_list) == 1:
        query_list = [query, query]
    return {
        **state,
        "intent": intent,
        "query_list": query_list[:2],
        "rewritten_query_1": query_list[1] if len(query_list) > 1 else query,
        "next_stage": "retrieve",
    }


# --- Stage 2: Retrieval & CRAG Evaluator ---
def stage2_retrieve(state: RAGState) -> RAGState:
    vs = _get_vector_store()
    docs = multi_query_retrieve(vs.vector_store, state["query_list"], k_per_query=2)
    return {**state, "documents": docs, "next_stage": "evaluate_crag"}


def stage2_evaluate_crag(state: RAGState) -> RAGState:
    query = state["query"]
    docs = state["documents"]
    if not docs:
        return {
            **state,
            "crag_score": 0.0,
            "crag_decision": "Incorrect",
            "need_web_search": True,
            "next_stage": "web_search",
        }
    #用于追踪
    sys.stdout.write("\n[终端] 阶段2 CRAG 评估中（Reranker 首次加载或计分可能较慢，请稍候）...\n")
    sys.stdout.flush()
    reranker = _get_reranker()
    doc_texts = [d.page_content for d in docs]
     # 原始 query + documents 与 rewrite_query_1 + documents 都评估，取最高分
    score_orig = reranker.crag_max_score(query, doc_texts)
    rew1 = state.get("rewritten_query_1") or query
    score_rew1 = reranker.crag_max_score(rew1, doc_texts)
    max_score = max(score_orig, score_rew1)
    sys.stdout.write("[终端] CRAG 计分完成，开始知识提炼...\n")
    sys.stdout.flush()
    if max_score > CRAG_CORRECT_THRESHOLD:
        decision = "Correct"
        need_web = False
        refined = refine_documents(query, docs)
        keep_docs = docs
    elif max_score >= CRAG_AMBIGUOUS_THRESHOLD:
        decision = "Ambiguous"
        need_web = True
        keep_docs = docs[:2]
        refined = refine_documents(query, keep_docs)
    else:
        decision = "Incorrect"
        need_web = True
        keep_docs = []
        refined = ""
    return {
        **state,
        "documents": keep_docs,
        "refined_context": refined,
        "crag_score": max_score,
        "crag_decision": decision,
        "need_web_search": need_web,
        "next_stage": "web_search" if need_web else "generate",
    }


# --- Stage 3: Web Search ---
def stage3_web_search(state: RAGState) -> RAGState:
    query_list = state["query_list"]
    web_results = web_search(query_list)
    decision = state.get("crag_decision", "Incorrect")
    refined = state.get("refined_context", "")
    # Path A: Incorrect -> documents = web_results only
    # Path B: Ambiguous (or backtrack) -> documents = refined + web_results
    if decision == "Incorrect":
        new_docs = web_results
        new_context = "\n\n".join(d.page_content for d in web_results)
    else:
        new_docs = state.get("documents", []) + web_results
        new_context = (refined + "\n\n" + "\n\n".join(d.page_content for d in web_results)) if web_results else refined
    return {
        **state,
        "web_results": web_results,
        "documents": new_docs,
        "refined_context": new_context.strip(),
        "need_web_search": False,
        "next_stage": "generate",
    }


# --- Stage 4: Generation ---
def stage4_generate(state: RAGState) -> RAGState:
    query = state["query"]
    context = state.get("refined_context") or ""
    if not context and state.get("documents"):
        context = "\n\n".join(d.page_content for d in state["documents"])
    prompt_text = load_generation_prompt()
    prompt = PromptTemplate.from_template(prompt_text)
    chain = prompt | chat_model | StrOutputParser()
    answer = chain.invoke({"query": query, "context": context or "(No reference documents.)"})
    return {
        **state,
        "answer": answer or "",
        "next_stage": "support_check",
    }


# --- Stage 5: Reflection — Support Check (hallucination) ---
def stage5_support_check(state: RAGState) -> RAGState:
    answer = state["answer"]
    docs = state.get("documents", [])
    doc_texts = [d.page_content for d in docs]
    if not doc_texts:
        return {**state, "support_scores": [], "next_stage": "utility_check"}
    reranker = _get_reranker()
    scores = reranker.support_scores(answer, doc_texts)
    supported = reranker.is_supported(answer, doc_texts, threshold=SUPPORT_THRESHOLD)
    halluci = state.get("halluci_generate_counter", 0)
    if not supported and halluci < MAX_HALLUCI_GENERATE:
        return {
            **state,
            "support_scores": scores,
            "next_stage": "generate",
            "halluci_generate_counter": halluci + 1,
        }
    return {**state, "support_scores": scores, "next_stage": "utility_check"}


# --- Stage 5: Utility Check ---
def stage5_utility_check(state: RAGState) -> RAGState:
    query = state["query"]
    answer = state["answer"]
    prompt_text = load_utility_check_prompt()
    prompt = PromptTemplate.from_template(prompt_text)
    chain = prompt | chat_model | StrOutputParser()
    out = chain.invoke({"query": query, "answer": answer})
    utility = "Useful" if out and "Useful" in (out or "") else "Not Useful"
    overall = state.get("overall_generate_counter", 0)
    if utility == "Useful":
        return {**state, "utility": utility, "final_answer": answer, "next_stage": "end"}
    if overall >= MAX_OVERALL_GENERATE:
        final = (answer or "") + "\n\n(已达最大重试次数，若仍不满意可重新提问。)"
        return {**state, "utility": utility, "final_answer": final, "next_stage": "end"}
    # Not Useful: rewrite query -> Rewritten_Query_2, compare with Rewritten_Query_1
    exp_prompt = load_query_expansion_prompt()
    exp_template = PromptTemplate.from_template(exp_prompt)
    raw = exp_template | chat_model | StrOutputParser()
    raw_str = raw.invoke({"query": query})
    try:
        obj = json.loads(raw_str)
        rew2 = (obj.get("queries") or [query])[0]
    except Exception:
        rew2 = query
    rew1 = state.get("rewritten_query_1", query)
    sim = query_similarity(rew1, rew2)
    if sim >= QUERY_SIM_THRESHOLD:
        # Path B: web search only, then generate
        web_results = web_search([query, rew2])
        refined = state.get("refined_context", "") + "\n\n" + "\n\n".join(d.page_content for d in web_results)
        return {
            **state,
            "utility": utility,
            "rewritten_query_2": rew2,
            "query_sim": sim,
            "overall_generate_counter": overall + 1,
            "documents": state.get("documents", []) + web_results,
            "refined_context": refined,
            "next_stage": "generate",
        }
    # Full loop: reset and go back to retrieve (with context management)
    return {
        **state,
        "utility": utility,
        "rewritten_query_2": rew2,
        "query_sim": sim,
        "overall_generate_counter": overall + 1,
        "query_list": [query, rew2],
        "rewritten_query_1": rew2,
        "next_stage": "retrieve",
    }


# --- Build graph ---
def build_rag_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("stage1_routing", stage1_routing)
    workflow.add_node("stage2_retrieve", stage2_retrieve)
    workflow.add_node("stage2_evaluate_crag", stage2_evaluate_crag)
    workflow.add_node("stage3_web_search", stage3_web_search)
    workflow.add_node("stage4_generate", stage4_generate)
    workflow.add_node("stage5_support_check", stage5_support_check)
    workflow.add_node("stage5_utility_check", stage5_utility_check)

    workflow.set_entry_point("stage1_routing")

    def after_routing(state: RAGState):
        n = state.get("next_stage", "retrieve")
        # ["stage2_retrieve", "stage4_generate"] -> 0 or 1
        if n == "generate":
            return "stage4_generate"
        return "stage2_retrieve"
        #return 1 if n == "generate" else 0

    def after_evaluate(state: RAGState):
        n = state.get("next_stage", "generate")
        # ["stage3_web_search", "stage4_generate"] -> 0 or 1
        if n == "web_search":
            return "stage3_web_search"
        return "stage4_generate"
        #return 0 if n == "web_search" else 1

    def after_support(state: RAGState):
        n = state.get("next_stage", "utility_check")
        # ["stage4_generate", "stage5_utility_check"] -> 0 or 1
        if n == "generate":
            return "stage4_generate"
        return "stage5_utility_check"
        #return 0 if n == "generate" else 1

    def after_utility(state: RAGState):
        n = state.get("next_stage", "end")
        if n == "end":
            return END
        if n == "generate":
            return "stage4_generate"
        return "stage2_retrieve"

    workflow.add_conditional_edges("stage1_routing", after_routing, ["stage2_retrieve", "stage4_generate"])
    workflow.add_edge("stage2_retrieve", "stage2_evaluate_crag")
    workflow.add_conditional_edges("stage2_evaluate_crag", after_evaluate, ["stage3_web_search", "stage4_generate"])
    workflow.add_edge("stage3_web_search", "stage4_generate")
    workflow.add_edge("stage4_generate", "stage5_support_check")
    workflow.add_conditional_edges("stage5_support_check", after_support, ["stage4_generate", "stage5_utility_check"])
    workflow.add_conditional_edges("stage5_utility_check", after_utility, [END, "stage4_generate", "stage2_retrieve"])

    return workflow.compile(checkpointer=MemorySaver())


def run_rag(query: str, thread_id: str | None = None) -> str:
    graph = build_rag_graph()
    config = {"configurable": {"thread_id": thread_id or "default"}}
    final_state = graph.invoke(_initial_state(query), config=config)
    return (final_state or {}).get("final_answer") or (final_state or {}).get("answer") or ""


def run_rag_stream(query: str, thread_id: str | None = None):
    """
    Run RAG graph with real-time workflow progress.
    Yields progress lines (当前阶段: ...) as each node runs, then yields the final answer.
    """
    graph = build_rag_graph()
    config = {"configurable": {"thread_id": thread_id or "default"}}
    initial: RAGState = _initial_state(query)
    sys.stdout.write(f"\n[终端] 当前 query: {query}\n")
    sys.stdout.flush()
    try:
        # stream_mode="updates" yields {node_name: state_update} per node（或部分版本为 tuple）
        for chunk in graph.stream(initial, config=config, stream_mode="updates"):
            node_name = None
            update = None
            if isinstance(chunk, dict):
                for k in chunk:
                    node_name = k
                    update = chunk[k]
                    break
            elif isinstance(chunk, (list, tuple)) and len(chunk) >= 2:
                node_name = chunk[0]
                update = chunk[1] if len(chunk) > 1 else {}
            if node_name:
                if update:
                    _print_stage_debug(node_name, update)
                label = STAGE_LABELS.get(node_name, node_name)
                yield f"▶ {label}\n"
        # 流结束后从 checkpoint 取最终状态（兼容 .values 或直接为 dict）
        state = graph.get_state(config)
        values = getattr(state, "values", state) if state else {}
        answer = (values or {}).get("final_answer") or (values or {}).get("answer") or ""
    except Exception as e:
        logger.exception("run_rag_stream error")
        answer = f"[执行出错: {e}]"
    yield "\n--- 回答 ---\n\n"
    for char in (answer or ""):
        yield char


def _initial_state(query: str) -> RAGState:
    return {
        "query": query,
        "query_list": [],
        "intent": "",
        "documents": [],
        "refined_context": "",
        "crag_score": 0.0,
        "crag_decision": "Incorrect",
        "need_web_search": False,
        "web_results": [],
        "answer": "",
        "support_scores": [],
        "halluci_generate_counter": 0,
        "overall_generate_counter": 0,
        "rewritten_query_1": "",
        "rewritten_query_2": "",
        "query_sim": 0.0,
        "utility": "",
        "final_answer": "",
        "next_stage": "",
    }


if __name__ == "__main__":
    out = run_rag("小户型适合哪些扫地机器人")

    print(out)
