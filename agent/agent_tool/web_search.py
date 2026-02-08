"""
Web search using Bright Data MCP (mcp_server.py).
Used in Stage 3: Correction & Web Search.
"""
import asyncio
from typing import List
from langchain_core.documents import Document
from utils.logger_handler import logger

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
    for q in query_list:
        if not (q and str(q).strip()):
            continue
        try:
           # Bright Data MCP 工具仅支持异步调用，用 ainvoke + _run_async 在同步上下文中执行
           async def _call_tool():
                try:
                    return await search_tool.ainvoke({"query": q})
                except (TypeError, KeyError):
                    return await search_tool.ainvoke({"input": q})
            result = _run_async(_call_tool())
        except Exception as e:
            logger.warning(f"[web_search] query '{q[:50]}...' failed: {e}")
            continue
        if result:
            text = result if isinstance(result, str) else str(result)
            docs.append(Document(page_content=text.strip(), metadata={"source": "web_search", "query": q}))
    return docs

