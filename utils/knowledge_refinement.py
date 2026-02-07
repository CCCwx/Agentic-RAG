"""
Knowledge refinement: extract core bullet points from retrieved documents for context.
Used after CRAG Correct/Ambiguous to compress documents before generation.
"""
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.prompt_loader import load_knowledge_refinement_prompt
from model.factory import chat_model


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
    chain = prompt | model | StrOutputParser()
    return chain.invoke({"query": query, "context": context})
