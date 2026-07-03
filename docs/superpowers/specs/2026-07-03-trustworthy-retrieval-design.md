# Trustworthy Retrieval Layer — Design Spec

**Date:** 2026-07-03
**Status:** Approved design, pending spec review
**Scope:** Retrieval quality gating, company disambiguation, explicit insufficient-evidence state propagated through the full agent pipeline, structurally-visible degradation in the frontend.

---

## 1. Problem

The RQ1/RQ2 evaluation exposed that retrieval can return evidence about the wrong
company and silently fall back to off-topic material while the pipeline still
produces a confident answer. For TSLA, all 5 retrieved evidence items were about
Meta, Intel, general tech, and Bybit crypto — and both the InvestIQ pipeline and
the monolithic baseline wrote confident bullish Tesla theses over them.

Three stacked root causes:

1. **Ingestion mislabels documents.** `scripts/fetch_news.py` queries NewsAPI with
   `q=<ticker>` and tags every returned article with that ticker. NewsAPI matches
   full text, so articles that mention the ticker in passing (a Bybit press release
   listing TSLA among seven contract symbols) are permanently labeled as Tesla
   evidence. The corpus is poisoned at write time.
2. **The query embedding carries no company signal.** `run_research` embeds only the
   generic question ("What are the key catalysts and risks?"); the ticker is only a
   Chroma metadata filter. Dense ranking never attempts company disambiguation — it
   trusts the poisoned label completely.
3. **No relevance gate and no evidence-quality state.** Whatever top-5 comes back is
   packaged as evidence regardless of distance. `ResearchOutputSchema` expresses
   quality only as a prose `summary` string. The analyst writes a confident thesis
   regardless; the sentiment agent independently retrieves and scored the off-topic
   chunks as "positive 0.60" for TSLA, which fed the risk score.

**Product decision (user):** when evidence is insufficient, produce a **degraded
report** — quantitative sections (trend, risk from market data) still run, the memo
carries an explicit evidence gap, and no document-based claims are made. Not a hard
refusal, not an auto-fetch.

---

## 2. Approach

**Chosen: query-time gates + cross-encoder re-ranking (no hybrid sparse channel).**

Retrieval pipeline: dense search over a widened candidate pool → deterministic
company-aboutness gate → cross-encoder relevance re-rank with threshold → typed
sufficiency status that downstream agents must respect.

### Considered and rejected

- **Gates only (no re-ranker):** raw bi-encoder cosine on all-MiniLM-L6-v2 is poorly
  calibrated for "is this passage relevant to this question"; a fixed cosine
  threshold either admits junk or rejects good evidence. Fixes disambiguation but
  only half-fixes relevance quality.
- **Hybrid dense+BM25 with RRF fusion:** BM25 solves a recall problem; the observed
  failure is precision. Ticker-filtered candidate pools are ~50–100 chunks, so a
  30-candidate dense fetch already covers most of the pool. A sparse index adds
  lifecycle complexity for near-zero gain at this corpus size. Revisit if the corpus
  grows ~100×.
- **LLM relevance judge:** ~1–2s + API cost per query, nondeterministic tests; the
  cross-encoder buys most of the benefit at a fraction of the cost.
- **NER at ingestion:** heavy dependency, brittle entity→ticker mapping. Alias
  matching is sufficient for company names in news text.
- **Wipe-and-re-ingest to clean the corpus:** NewsAPI free tier reaches back only
  ~30 days; re-fetching shrinks the corpus. Query-time gates defend against legacy
  poison without a migration.
- **Forced "watch" action on insufficient evidence:** rejected by user. "Watch" is a
  pseudo-recommendation. Insufficient evidence yields a genuine **no-view** state
  (see §5.4), coherent with the upcoming verdict reframe.

**Locked decisions (user-confirmed, do not revisit):** cross-encoder choice
(`cross-encoder/ms-marco-MiniLM-L-6-v2`), candidate pool size (30), query-time
gating (no corpus migration).

---

## 3. New retrieval pipeline

### 3.1 `core/company_registry.py` (new)

Resolves ticker → company name + aliases. Resolution order:

1. `COMPANY_ALIASES` override map in `config.py` (seeded with the 5 eval tickers).
2. JSON cache at `data/company_aliases.json`.
3. yfinance `Ticker.info` `shortName`/`longName` (already a project dependency;
   the trend agent hits yfinance for the same ticker anyway). Corporate suffixes
   ("Inc", "Corp", "Corporation", "Ltd", ",") stripped to produce clean aliases
   (e.g. `TSLA → ["Tesla", "Tesla, Inc."]`). Result written to the JSON cache.
4. Offline/failure fallback: `[<ticker>]` alone, flagged so retrieval can note
   reduced disambiguation power in `status_reason`.

Public interface: `get_company(ticker) -> CompanyInfo` where `CompanyInfo` is a
small Pydantic model `{ticker: str, name: str, aliases: list[str], source:
Literal["config","cache","yfinance","fallback"]}`.

### 3.2 `core/retrieval.py` (new) — retrieval policy

`VectorStoreManager` stays a thin Chroma wrapper; its `query()` API is unchanged.
All policy lives in a new module with one entry point:

```
retrieve_evidence(
    ticker: str,
    question: str,
    days_back: Optional[int],
    top_k: int = 5,
    store: Optional[VectorStoreManager] = None,   # singleton default
) -> RetrievalResult
```

Pipeline stages:

1. **Company-aware query text:** `f"{company.name} ({ticker}): {question}"` — the
   embedding finally carries a company signal.
2. **Dense search:** fetch `RETRIEVAL_FETCH_N = 30` candidates with the existing
   Chroma ticker prefilter (kept as a useful narrowing filter, no longer trusted
   as ground truth).
3. **Aboutness gate:** pure function
   `aboutness_score(text: str, company: CompanyInfo) -> float` in `[0, 1]`,
   deterministic, based on word-boundary mention counts of ticker and aliases
   (case-insensitive for names, case-sensitive for the ticker symbol).
   Properties: 0.0 when nothing matches; monotonically nondecreasing in mention
   count; saturates at 1.0. Exact formula tuned during calibration (§7).
   Effective score per candidate = `max(chunk-level score, article-level score
   from ingestion metadata when present)` — protects mid-article chunks of
   genuinely on-topic pieces that use pronouns instead of the company name.
   Candidates below `ABOUTNESS_THRESHOLD` are dropped.
4. **Cross-encoder re-rank:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (~22MB), loaded
   lazily as a singleton via `core/singletons.get_reranker()`. Scores
   (query, passage) pairs jointly for survivors only; candidates below
   `RERANK_THRESHOLD` are dropped; top `top_k` survivors returned, ordered by
   re-rank score. If the model cannot load, fall back to aboutness + cosine
   ordering, cap status at `partial`, and record the reason (mirrors the
   FinBERT→VADER fallback pattern).
5. **Time preference:** keep the try-recent-first behavior, but a fallback to
   evidence older than `days_back` is no longer silent — it caps status at
   `partial` with the reason recorded.
6. **Result:** `RetrievalResult` carrying gated evidence with per-item scores,
   plus:
   - `evidence_status`: `"sufficient"` (≥ `MIN_SUFFICIENT_EVIDENCE = 3` items
     passed all gates, time-filtered, full gate stack ran) · `"partial"` (1–2
     items passed, or stale fallback used, or re-ranker unavailable) ·
     `"insufficient"` (0 items passed).
   - `status_reason`: human-readable accounting, e.g. *"30 candidates retrieved;
     24 rejected by aboutness gate, 4 by relevance threshold; 2 passed —
     partial evidence."*

Latency budget: +~0.3–0.5s per query on CPU (re-rank ≤30 pairs, aboutness is
string matching), against a ~20s pipeline. The sentiment refactor (§5.2) removes
one redundant embedding query, partially offsetting this.

### 3.3 Config additions (`config.py`)

```
RETRIEVAL_FETCH_N = 30
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ABOUTNESS_THRESHOLD = <calibrated, §7>
RERANK_THRESHOLD = <calibrated, §7>
MIN_SUFFICIENT_EVIDENCE = 3
COMPANY_ALIASES: dict[str, list[str]]  # manual overrides, seeded for eval tickers
```

Calibrated values are committed with a comment recording the calibration date and
measured false-reject rate.

---

## 4. Schema changes — all additive, defaults preserve backward compatibility

| Schema | New field | Default |
|---|---|---|
| `EvidenceSchema` | `relevance_score: Optional[float]` (cross-encoder) | `None` |
| `EvidenceSchema` | `aboutness_score: Optional[float]` | `None` |
| `ResearchOutputSchema` | `evidence_status: Literal["sufficient","partial","insufficient"]` | `"sufficient"` |
| `ResearchOutputSchema` | `status_reason: str` | `""` |
| `SentimentOutputSchema` | `data_status: Literal["ok","no_data"]` | `"ok"` |
| `ActionSignalSchema.signal` | add `"no_view"` to the Literal | n/a |
| `InvestmentMemoSchema` | `evidence_status: Literal["sufficient","partial","insufficient"]` | `"sufficient"` |
| `InvestmentMemoSchema` | `debate_skipped_reason: Optional[str]` | `None` |

`similarity_score` is kept on `EvidenceSchema`. No existing field changes type or
meaning. All existing consumers of `memo.action.signal` (memory agent, main.py,
coordinator trace, evaluation scripts) treat it as a string and flow correctly
with the new `"no_view"` value — this is why no-view is a literal value rather
than `action: Optional[...]`, which would crash six call sites.

---

## 5. Downstream propagation — refusing to fabricate

### 5.1 Research agent (`agents/research_agent.py`)

Delegates to `core.retrieval.retrieve_evidence()`; maps `RetrievalResult` into
`ResearchOutputSchema` including `evidence_status`, `status_reason`, and per-item
scores. Public signature unchanged.

### 5.2 Sentiment agent (`agents/sentiment_agent.py`)

Gains `research: Optional[ResearchOutputSchema] = None`. When provided (the
coordinator path), it scores that evidence pack directly — its docstring already
claims this; now it will be true — and no longer runs its own retrieval
(removes a redundant embedding query per run). When omitted (standalone use),
it runs research itself as today.

When `evidence_status == "insufficient"` or the evidence list is empty:
`data_status="no_data"`, `overall_score=0.0`, `overall_label="neutral"`,
`items=[]`, summary states there is nothing to score. FinBERT is never invoked
on off-topic text.

### 5.3 Risk agent (`agents/risk_agent.py`)

When `sentiment.data_status == "no_data"`: the sentiment component is excluded
from the risk score (price/volatility components reweighted, exact reweighting
decided at implementation against the current scoring code), and an
informational flag is added: *"No sentiment data — insufficient evidence."*
Fabricated neutrality must not be treated as signal.

### 5.4 Analyst agent (`agents/analyst_agent.py`)

Behavior branches on `research.evidence_status`:

- **`insufficient`:** degraded memo. The recommendation LLM call is *not made*.
  `action = ActionSignalSchema(signal="no_view", confidence=0.0, rationale=
  "No trustworthy evidence retrieved for <ticker>; declining to take a view.")`
  Thesis is written from market data only (trend/risk), opens with the evidence
  gap, and is prompted to make no document-based claims; deterministic fallback
  exists as today. `catalysts=[]`; `risks` come from risk flags only;
  `citations=[]`. This is a genuine absence-of-view state, not a soft
  recommendation — coherent with the planned verdict reframe.
- **`partial`:** memo is generated normally but the thesis must note the limited
  evidence base, and action confidence is capped at `0.6`.
- **`sufficient`:** current behavior.

`memo.evidence_status` is set from research in all branches.

### 5.5 Debate agent / coordinator (`agents/coordinator_agent.py`)

On `insufficient`, debate is **skipped** (there is no evidence to debate):

- `memo.debate_skipped_reason = "insufficient evidence"` — visible in the report
  itself, not only the trace (user requirement).
- Pipeline trace records it.
- The streaming generator emits `{"event": "skipped", "agent": "debate",
  "message": "insufficient evidence"}` (new event type; unknown types are
  already ignored by the frontend handler, so this is backward-safe).

Both `_run_pipeline` and `stream_pipeline_events` are updated (they duplicate
the wiring); research → sentiment now passes the research object through.

### 5.6 Memory agent — unchanged

`no_view` flows through `action_signal: str` and signal-change comparison
correctly with no code change.

---

## 6. Ingestion fix (`scripts/fetch_news.py`)

- NewsAPI query becomes `q='"<company name>" OR "<ticker>"'` via the company
  registry, instead of the bare ticker string.
- At ingest, compute `aboutness_score` on the **full article** (title + description
  + content, title-weighted). Articles below `ABOUTNESS_THRESHOLD` are skipped
  entirely — passing mentions never enter the corpus.
- Chunks store `about_score` (article-level) in Chroma metadata, which the
  query-time gate uses via the `max(chunk, article)` rule (§3.2 step 3).
- Legacy documents (no `about_score` metadata) are handled by the query-time
  chunk-level gate; no migration.

---

## 7. Threshold calibration (`evaluation/calibrate_retrieval.py`, new)

A committed script (matching the project's evaluation culture) that runs the
retrieval stages **without gates** over the live corpus and reports per-candidate
`(aboutness, cosine similarity, cross-encoder score)` for:

- **Labeled negatives:** the 5 TSLA off-topic items from the RQ1 eval
  (Meta, Intel, tech roundup, 2× Bybit).
- **Labeled positives:** manually verified on-topic evidence for AAPL and NVDA
  (the tickers where retrieval was genuinely good), plus MSFT/GOOGL as additional
  positives.

Outputs a score table and, for any candidate `(ABOUTNESS_THRESHOLD,
RERANK_THRESHOLD)` pair, the resulting confusion: negatives admitted and
**positives falsely rejected**.

**Acceptance criteria (user requirement):**

1. All 5 TSLA negatives rejected.
2. False-reject rate on verified positives **reported explicitly**, target 0.
3. If rejecting all negatives forces FRR > 0 on positives, the tradeoff is
   surfaced to the user before thresholds are locked — the gate must not be
   tuned strict-by-default at the cost of good evidence.

Chosen values are committed to `config.py` with calibration date and measured FRR
in a comment. Post-fix, re-running retrieval for TSLA must yield
`insufficient`/`partial` with zero Meta/Intel/Bybit items passing — this is the
end-to-end acceptance test.

---

## 8. Frontend (`frontend/index.html`) — structurally visible degradation

Degradation is a **rendering mode**, not a banner (user requirement: weak- and
strong-evidence reports must be impossible to confuse at a glance).

The results panel carries `data-evidence-status="sufficient|partial|insufficient"`
driving CSS. Status is read from the streamed `research` event as soon as it
arrives (before the analyst finishes), and from `memo.evidence_status` on the
final render.

**Insufficient mode:**

- **Dominant gap notice** rendered as the first card, above the hero: large,
  amber/red-accented block stating that no trustworthy evidence was found, what
  was rejected (from `status_reason`), and that only market-data analysis
  follows.
- **Hero:** action badge renders a distinct muted "NO VIEW" state (new
  `action-no_view` style — grey, not a colored recommendation chip). The
  "Confidence: 0%" line is replaced by "No view taken — insufficient evidence"
  (a 0% confidence readout looks like a bug, not a decision).
- **Recommendation card:** replaced by the no-view explanation; the reco-reasons
  list shows only market-data reasons.
- **Analysis cards** (trend, risk): visually muted (reduced opacity, "market
  data only" tag) — still readable, clearly subordinate to the gap notice.
- **Sentiment card:** when `data_status === "no_data"`, **no gauge is drawn**.
  An explicit empty state ("No sentiment data — no relevant evidence") replaces
  the chart. The current behavior (neutral-looking dial at 0.00) is exactly the
  confusable state this redesign eliminates.
- **Debate card:** renders `debate_skipped_reason` ("Debate skipped:
  insufficient evidence").
- **Evidence card:** "No trustworthy evidence passed retrieval gates" plus the
  rejection accounting from `status_reason`.

**Partial mode:** gap notice styled as a warning (present but less dominant),
analysis cards not muted, evidence table shows per-item `relevance_score` /
`aboutness_score` columns, confidence visibly capped.

**Sufficient mode:** current rendering, plus the new score columns in the
evidence table.

**Legacy graceful degradation (user requirement, confirmed):** every new field is
read with a fallback (`research?.evidence_status ?? null`, etc.). A payload
without `evidence_status` (old saved report, older API) renders exactly as today:
no degraded styling, no gap notice, no errors. The existing renderers already
default defensively (`memo.action || {}`, `sentiment.overall_score || 0`);
new code follows the same pattern. Unknown stream event types are already
ignored. The `sig → class` map gains `no_view`; unknown signals still fall back
to the hold style as today.

---

## 9. Testing (TDD, as always)

New test files (cross-encoder and yfinance mocked everywhere; existing 234 tests
must stay green):

| File | Covers |
|---|---|
| `tests/test_company_registry.py` | config override → cache → yfinance mock → offline fallback; suffix stripping; cache writes |
| `tests/test_aboutness.py` | pure-function properties (zero/monotonic/saturation, word boundaries); fixtures from the real TSLA failures: Bybit chunk scores low, genuine Tesla text scores high |
| `tests/test_retrieval_pipeline.py` | company-aware query text; fetch_n; gate ordering; status rules (≥3 sufficient, 1–2 partial, 0 insufficient, stale-fallback→partial, no-reranker→partial); `status_reason` accounting; injected fake re-ranker |
| `tests/test_research_agent_status.py` | schema population, per-item scores, backward-compatible defaults |
| `tests/test_sentiment_evidence.py` | consumes provided research pack (no store query); `no_data` on insufficient/empty; standalone path unchanged |
| `tests/test_risk_no_sentiment.py` | sentiment excluded from score on `no_data`; informational flag |
| `tests/test_analyst_degraded.py` | `no_view`/0.0 on insufficient with no recommendation LLM call; 0.6 confidence cap on partial; market-data-only thesis prompt; `evidence_status` on memo |
| `tests/test_coordinator_propagation.py` | end-to-end wiring with injected fns: insufficient → sentiment no_data → analyst no_view → debate skipped with visible reason → stream emits `skipped` event and statuses |
| `tests/test_fetch_news_aboutness.py` | company-name query construction; below-floor articles skipped; `about_score` metadata on chunks |
| `tests/test_retrieval_integration.py` | one slow-marked test with the real cross-encoder |

Frontend has no JS test harness (consistent with repo practice): manual
verification checklist — a forced-insufficient run (nonsense-corpus ticker), a
partial run, a sufficient run, and a legacy payload replay — verified via the
running app before completion.

---

## 10. Implementation order

1. Company registry (+tests)
2. Aboutness function (+tests)
3. Retrieval pipeline module (+tests)
4. Schema additions (+tests)
5. Research agent delegation (+tests)
6. Sentiment refactor (+tests)
7. Risk exclusion (+tests)
8. Analyst degraded/no-view behavior (+tests)
9. Coordinator + streaming propagation, debate skip (+tests)
10. fetch_news ingestion fix (+tests)
11. Frontend degraded rendering + legacy fallback
12. Calibration script → run against live corpus → report FRR to user → lock
    thresholds in config
13. Acceptance: re-run TSLA retrieval (expect insufficient/partial, zero
    off-topic items); full test suite; update CLAUDE.md checklist (Fix 9)

## 11. Accepted tradeoffs

- +~0.3–0.5s query latency and one more ~22MB model in memory (partially offset
  by removing the sentiment agent's redundant retrieval).
- Stricter gates mean some tickers that previously "worked" will honestly report
  partial/insufficient — more no-view outcomes until the corpus improves. This
  is the point.
- Alias matching can miss chunks that reference the company only obliquely
  ("the EV maker"); mitigated by the article-level `about_score` for new
  ingests and by the cross-encoder (whose query names the company), and
  measured by the FRR calibration requirement.
