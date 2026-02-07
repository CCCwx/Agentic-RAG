"""
CRAG Evaluator & Hallucination Support Check using bge-reranker.
- Stage 2: score (Query, Retrieved Doc) -> CRAG decision.
- Stage 5: score (Generated Answer, Document) -> support check (high = supported, 0 = likely hallucination).
"""
from typing import List, Tuple
import math
from utils.logger_handler import logger

try:
    from FlagEmbedding import FlagReranker
except ImportError:
    FlagReranker = None


def _sigmoid(x: float) -> float:
    """Map raw score to [0, 1] for threshold comparison."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


class RerankerService:
    """Wrapper for BAAI/bge-reranker: score query-document or answer-document pairs."""

    def __init__(self, model_name: str | None = None):
        if model_name is None:
            try:
                from utils.config_handler import rag_conf
                model_name = rag_conf.get("reranker_model_name", "BAAI/bge-reranker-base")
            except Exception:
                model_name = "BAAI/bge-reranker-base"
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if FlagReranker is None:
            raise ImportError("Install FlagEmbedding: pip install FlagEmbedding")
        if self._model is None:
            self._model = FlagReranker(
                self.model_name,
                use_fp16=True,
            )
        return self._model

    def compute_scores(
        self,
        pairs: List[Tuple[str, str]],
        normalize: bool = True,
    ) -> List[float]:
        """
        Score each (query, document) or (answer, document) pair.
        :param pairs: list of (text_a, text_b) e.g. (query, doc) or (answer, doc).
        :param normalize: if True, map raw scores to [0,1] via sigmoid for CRAG thresholds.
        :return: list of scores, same order as pairs.
        """
        if not pairs:
            return []
        # FlagReranker.compute_score expects list of [query, doc] or list of pairs
        list_pairs = [[a, b] for a, b in pairs]
        raw = self.model.compute_score(list_pairs)
        if isinstance(raw, (int, float)):
            raw = [raw]
        if normalize:
            return [_sigmoid(float(s)) for s in raw]
        return [float(s) for s in raw]

    def crag_max_score(self, query: str, doc_texts: List[str]) -> float:
        """
        Stage 2 CRAG: max score over (query, doc) for each doc.
        :return: max of normalized scores in [0,1].
        """
        if not doc_texts:
            return 0.0
        pairs = [(query, d) for d in doc_texts]
        scores = self.compute_scores(pairs, normalize=True)
        return max(scores)

    def support_scores(self, answer: str, doc_texts: List[str]) -> List[float]:
        """
        Stage 5 hallucination check: (answer, doc) -> score per doc.
        High score = document supports answer; score ~0 = likely hallucination for that doc.
        :return: list of normalized scores, one per doc.
        """
        if not doc_texts:
            return []
        pairs = [(answer, d) for d in doc_texts]
        return self.compute_scores(pairs, normalize=True)

    def is_supported(self, answer: str, doc_texts: List[str], threshold: float = 0.3) -> bool:
        """
        Answer is supported if at least one (answer, doc) score >= threshold.
        If all scores are 0 or very low, treat as not supported (hallucination).
        """
        scores = self.support_scores(answer, doc_texts)
        if not scores:
            return False
        return max(scores) >= threshold
