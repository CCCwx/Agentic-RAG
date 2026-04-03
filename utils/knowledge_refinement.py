"""
Knowledge refinement: extract core bullet points from retrieved documents for context.
Used after CRAG Correct/Ambiguous to compress documents before generation.
"""
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from utils.prompt_loader import load_knowledge_refinement_prompt
from model.factory import chat_model
from schema.schema import RefinedBulletsResponse
from utils.structured_call import invoke_structured_with_retry


'''
先把所有 docs 拼成一个大 context（类似 [Doc1]... [Doc2]...）
然后 调用一次 LLM，让它从“整个 context”里提炼 bullet points
'''
def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[Doc {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )


def refine_documents(
    query: str,
    docs: list[Document],
    model=None,
) -> str:
    """
    Extract key bullet points from documents relevant to the query.
    :return: Single string of bullet points (refined context).
    """
    if not docs:
        return ""
    if model is None:
        model = chat_model
    template = load_knowledge_refinement_prompt()
    prompt = PromptTemplate.from_template(template)
    context = _format_docs(docs)
    inputs = {"query": query, "context": context}

    def _fallback() -> RefinedBulletsResponse:
        # If structured parsing keeps failing, return a minimal safe bullet.
        # Avoid polluting context with unstructured text.
        return RefinedBulletsResponse(bullets=["材料不足或提炼失败"])  # type: ignore[arg-type]

    result = invoke_structured_with_retry(
        prompt=prompt,
        model=model,
        schema=RefinedBulletsResponse,
        inputs=inputs,
        retries=1,
        fallback=_fallback,
    )
    bullets = result.bullets if result and getattr(result, "bullets", None) else []

    cleaned: list[str] = []
    seen = set()
    for b in bullets:
        if not b:
            continue
        s = str(b).strip()
        if not s:
            continue
        # remove leading bullet markers if model included them
        if s.startswith("- "):
            s = s[2:].strip()
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s)

    # Render to stable bullet-point string for downstream stages
    return "\n".join(f"- {s}" for s in cleaned) if cleaned else ""

