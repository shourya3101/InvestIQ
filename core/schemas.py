"""
Pydantic schemas for the URECA research system.

Provides strict data validation for documents, evidence items,
and research output before scaling to multi-agent reasoning.
"""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class DocumentSchema(BaseModel):
    """Schema for a document ingested into the vector store."""

    content: str
    source: str
    ticker: Optional[str] = None
    date: Optional[datetime] = None
    filepath: str


class EvidenceSchema(BaseModel):
    """Schema for a single piece of retrieved evidence."""

    citation_id: str
    snippet: str
    filepath: str
    source: str
    ticker: Optional[str] = None
    date: Optional[datetime] = None
    similarity_score: float


class ResearchOutputSchema(BaseModel):
    """Schema for the full output of a research query."""

    ticker: str
    question: str
    days_back: Optional[int] = None
    evidence: List[EvidenceSchema] = Field(default_factory=list)
    summary: str


# ── Trend agent schemas ──────────────────────────────────────────────────


class TrendSignalSchema(BaseModel):
    """Schema for a single trend signal over a given horizon."""

    horizon: Literal["7d", "30d", "90d"]
    return_pct: float
    volatility_pct: float
    max_drawdown_pct: float
    trend_label: Literal["bullish", "neutral", "bearish"]


class TrendOutputSchema(BaseModel):
    """Schema for the full output of a trend analysis."""

    ticker: str
    mode: str  # "live" or "offline"
    as_of: datetime
    signals: List[TrendSignalSchema] = Field(default_factory=list)
    summary: str


# ── Sentiment agent schemas ──────────────────────────────────────────────


class SentimentItemSchema(BaseModel):
    """Schema for a single document-level sentiment score."""

    citation_id: str
    polarity: float  # -1 to +1
    label: Literal["positive", "neutral", "negative"]
    rationale: str  # short explanation (keywords matched or score)
    date: Optional[datetime] = None
    filepath: str


class SentimentOutputSchema(BaseModel):
    """Schema for the full output of a sentiment analysis."""

    ticker: str
    as_of: datetime
    window_days: int
    overall_score: float
    overall_label: Literal["positive", "neutral", "negative"]
    items: List[SentimentItemSchema] = Field(default_factory=list)
    summary: str


# ── Risk agent schemas ──────────────────────────────────────────────────


class RiskFlagSchema(BaseModel):
    """Schema for a single risk flag raised by the Risk agent."""

    category: Literal["price", "volatility", "sentiment"]
    severity: Literal["low", "moderate", "high"]
    message: str


class RiskOutputSchema(BaseModel):
    """Schema for the full output of a risk assessment."""

    ticker: str
    as_of: datetime
    risk_score: float  # 0-100
    risk_level: Literal["low", "moderate", "high"]
    flags: List[RiskFlagSchema] = Field(default_factory=list)
    summary: str


# ── Investment memo schemas ─────────────────────────────────────────────


class ActionSignalSchema(BaseModel):
    """Schema for a single action signal with confidence."""

    signal: Literal["buy", "hold", "sell", "watch"]
    confidence: float  # 0-1
    rationale: str


class InvestmentMemoSchema(BaseModel):
    """Schema for the full investment memo output."""

    ticker: str
    as_of: datetime
    question: str
    thesis: str
    catalysts: List[str]
    risks: List[str]
    action: ActionSignalSchema
    citations: List[str]  # e.g., ["E1", "E3"]
    risk_level: str
    risk_score: float
    writer_mode: str  # "deterministic" | "groq" | "claude" | "auto"
    debate: Optional["DebateOutputSchema"] = None
    memory: Optional["MemoryComparisonSchema"] = None


# ─── Debate Agent Schemas ───────────────────────────────


class DebateArgumentSchema(BaseModel):
    side: Literal["bull", "bear"]
    arguments: List[str]
    confidence: float
    key_evidence: List[str]


class DebateOutputSchema(BaseModel):
    ticker: str
    as_of: datetime
    bull: DebateArgumentSchema
    bear: DebateArgumentSchema
    coordinator_verdict: str
    final_bias: Literal["bullish", "bearish", "neutral"]
    memo_update: str


# ─── Memory Agent Schemas ────────────────────────────────


class MemoHistoryEntrySchema(BaseModel):
    ticker: str
    as_of: datetime
    risk_score: float
    risk_level: str
    action_signal: str
    confidence: float
    thesis_snippet: str
    memo_hash: str


class MemoryComparisonSchema(BaseModel):
    ticker: str
    current_as_of: datetime
    previous_as_of: Optional[datetime] = None
    risk_score_delta: Optional[float] = None
    signal_changed: bool
    thesis_changed: bool
    summary: str


# ─── Coordinator / Full Analysis Schema ──────────────────


class FullAnalysisSchema(BaseModel):
    ticker: str
    mode: str
    as_of: datetime
    memo: InvestmentMemoSchema
    research: ResearchOutputSchema
    trend: TrendOutputSchema
    sentiment: SentimentOutputSchema
    risk: RiskOutputSchema
    debate: Optional[DebateOutputSchema] = None
    memory: Optional[MemoryComparisonSchema] = None
    pipeline_trace: List[str]
    total_runtime_seconds: float


if __name__ == "__main__":
    """Smoke-test: instantiate dummy schemas and print them."""

    doc = DocumentSchema(
        content="AAPL reported record quarterly revenue of $130 billion.",
        source="bloomberg_export",
        ticker="AAPL",
        date=datetime(2026, 1, 15),
        filepath="sample_document.txt",
    )
    print("── DocumentSchema ──")
    print(doc.model_dump_json(indent=2))

    trend = TrendOutputSchema(
        ticker="AAPL",
        mode="offline",
        as_of=datetime(2026, 2, 20),
        signals=[
            TrendSignalSchema(
                horizon="7d",
                return_pct=1.25,
                volatility_pct=18.4,
                max_drawdown_pct=-2.1,
                trend_label="bullish",
            ),
            TrendSignalSchema(
                horizon="30d",
                return_pct=-0.8,
                volatility_pct=22.1,
                max_drawdown_pct=-5.3,
                trend_label="neutral",
            ),
            TrendSignalSchema(
                horizon="90d",
                return_pct=-4.5,
                volatility_pct=27.0,
                max_drawdown_pct=-12.0,
                trend_label="bearish",
            ),
        ],
        summary="Short-term bullish momentum; medium/long-term caution.",
    )
    print("\n── TrendOutputSchema ──")
    print(trend.model_dump_json(indent=2))

    sentiment = SentimentOutputSchema(
        ticker="AAPL",
        as_of=datetime(2026, 2, 20),
        window_days=30,
        overall_score=0.35,
        overall_label="positive",
        items=[
            SentimentItemSchema(
                citation_id="sent-1",
                polarity=0.6,
                label="positive",
                rationale="strong revenue growth keywords",
                date=datetime(2026, 1, 15),
                filepath="sample_document.txt",
            ),
            SentimentItemSchema(
                citation_id="sent-2",
                polarity=-0.2,
                label="negative",
                rationale="supply-chain risk mention",
                date=datetime(2026, 2, 1),
                filepath="report_q4.txt",
            ),
        ],
        summary="Mildly positive sentiment; strong revenue offset by supply-chain concerns.",
    )
    print("\n── SentimentOutputSchema ──")
    print(sentiment.model_dump_json(indent=2))

    risk = RiskOutputSchema(
        ticker="AAPL",
        as_of=datetime(2026, 2, 20),
        risk_score=42.0,
        risk_level="moderate",
        flags=[
            RiskFlagSchema(
                category="price",
                severity="moderate",
                message="30d return near zero; sideways drift.",
            ),
            RiskFlagSchema(
                category="volatility",
                severity="high",
                message="Annualised vol exceeds 25%.",
            ),
            RiskFlagSchema(
                category="sentiment",
                severity="low",
                message="Mildly positive sentiment (0.35).",
            ),
        ],
        summary="Moderate overall risk: elevated volatility offset by stable sentiment.",
    )
    print("\n── RiskOutputSchema ──")
    print(risk.model_dump_json(indent=2))

    memo = InvestmentMemoSchema(
        ticker="AAPL",
        as_of=datetime(2026, 2, 20),
        question="What are the key catalysts and risks?",
        thesis="Strong earnings momentum with manageable supply-chain headwinds.",
        catalysts=["Record Q4 revenue", "AI product pipeline"],
        risks=["Supply-chain disruption", "Elevated short-term volatility"],
        action=ActionSignalSchema(
            signal="hold",
            confidence=0.72,
            rationale="Positive fundamentals offset by near-term uncertainty.",
        ),
        citations=["E1", "E2", "E3"],
        risk_level="moderate",
        risk_score=42.0,
        writer_mode="deterministic",
    )
    print("\n── InvestmentMemoSchema ──")
    print(memo.model_dump_json(indent=2))
