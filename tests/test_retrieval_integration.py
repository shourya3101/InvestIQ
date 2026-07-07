"""Real CrossEncoder smoke test — marked slow, excluded from default dev loop."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from core.company_registry import CompanyInfo
from core.retrieval import retrieve_evidence

TSLA = CompanyInfo(ticker="TSLA", name="Tesla", aliases=["Tesla"], source="config")


@pytest.mark.slow
def test_real_cross_encoder_separates_on_topic_from_off_topic():
    from sentence_transformers import CrossEncoder
    from config import RERANK_MODEL_NAME
    reranker = CrossEncoder(RERANK_MODEL_NAME)

    query = "Tesla (TSLA): What are the key catalysts and risks?"
    on_topic = "Tesla reported record quarterly deliveries and raised its production guidance."
    off_topic = "Bybit expands perpetual contracts to include seven new TradFi assets."
    scores = reranker.predict([(query, on_topic), (query, off_topic)])
    assert scores[0] > scores[1]
