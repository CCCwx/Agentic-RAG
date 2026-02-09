"""
Web search using Bright Data MCP (mcp_server.py).
Used in Stage 3: Correction & Web Search.
"""
import asyncio
import os
import sys
from typing import List
from langchain_core.documents import Document
from utils.logger_handler import logger

# Bright Data 代理可能较慢，延长单次搜索等待时间（秒）；可通过环境变量 WEB_SEARCH_TIMEOUT 覆盖
WEB_SEARCH_TIMEOUT_SECONDS = int(os.environ.get("WEB_SEARCH_TIMEOUT", "120"))

# Lazy cache of MCP tools
_mcp_tools = None
_mcp_tools_lock = asyncio.Lock()


async def _get_mcp_tools():
    global _mcp_tools
    async with _mcp_tools_lock:
        if _mcp_tools is not None:
            return _mcp_tools
        try:
            from mcp_server import setup_bright_data_tools
            _mcp_tools = await setup_bright_data_tools()
            return _mcp_tools
        except Exception as e:
            logger.error(f"[web_search] MCP setup failed: {e}")
            return []


def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass
    return asyncio.run(coro)


def _find_search_tool(tools):
    """Prefer tool named web_search, search, or first tool that accepts query."""
    for t in tools:
        name = getattr(t, "name", "") or ""
        if "search" in name.lower() or "web" in name.lower():
            return t
    return tools[0] if tools else None


def web_search(query_list: List[str]) -> List[Document]:
    """
    Run web search for each query in query_list using Bright Data MCP.
    :return: List of Document (one per query result block) for downstream context.
    """
    if not query_list:
        return []
    tools = _run_async(_get_mcp_tools())
    search_tool = _find_search_tool(tools)
    if not search_tool:
        logger.warning("[web_search] No search tool found from MCP")
        return []
    docs = []
    total = len([q for q in query_list if q and str(q).strip()])
    sys.stdout.write(f"\n[终端] 阶段3 网络搜索开始（共 {total} 条，单条最多等待 {WEB_SEARCH_TIMEOUT_SECONDS}s）...\n")
    sys.stdout.flush()
    idx = 0
    for q in query_list:
        if not (q and str(q).strip()):
            continue
        idx += 1
        short_q = (q[:40] + "…") if len(q) > 40 else q
        sys.stdout.write(f"[终端] 正在搜索 ({idx}/{total}): {short_q}\n")
        sys.stdout.flush()
        try:
            async def _call_tool(qq=q):
                try:
                    return await search_tool.ainvoke({"query": qq})
                except (TypeError, KeyError):
                    return await search_tool.ainvoke({"input": qq})
            result = _run_async(
                asyncio.wait_for(_call_tool(), timeout=WEB_SEARCH_TIMEOUT_SECONDS)
            )
        except asyncio.TimeoutError:
            sys.stdout.write(f"[终端] 本条搜索超时（{WEB_SEARCH_TIMEOUT_SECONDS}s），跳过\n")
            sys.stdout.flush()
            logger.warning(
                f"[web_search] query '{q[:50]}...' timed out after {WEB_SEARCH_TIMEOUT_SECONDS}s"
            )
            continue
        except Exception as e:
            sys.stdout.write(f"[终端] 本条搜索失败: {e}\n")
            sys.stdout.flush()
            logger.warning(f"[web_search] query '{q[:50]}...' failed: {e}")
            continue
        sys.stdout.write("[终端] 本条搜索完成\n")
        sys.stdout.flush()
        if result:
            text = result if isinstance(result, str) else str(result)
            docs.append(Document(page_content=text.strip(), metadata={"source": "web_search", "query": q}))
    sys.stdout.write(f"[终端] 阶段3 网络搜索结束，得到 {len(docs)} 条结果\n")
    sys.stdout.flush()
    return docs

