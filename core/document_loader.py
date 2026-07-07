"""
Document Loader for the URECA research system.

Loads TXT and CSV files, returning validated DocumentSchema objects.
Long documents are split into overlapping 800-character windows so that
retrieval quality stays high even for multi-page Bloomberg exports.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

from core.schemas import DocumentSchema

# ── Chunking constants ────────────────────────────────────────────────────────

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


# ── Sliding-window chunker ────────────────────────────────────────────────────


def chunk_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split *text* into overlapping windows of *size* characters.

    Adjacent windows share *overlap* characters so that sentences that fall
    on a boundary appear in full in at least one chunk.

    Returns an empty list for empty input; a single-element list when
    ``len(text) <= size``.
    """
    if not text:
        return []
    if len(text) <= size:
        return [text]
    step = size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


def _extract_ticker_and_date_from_text(
    text: str,
) -> Tuple[Optional[str], Optional[datetime]]:
    """
    Scan *text* for ``Ticker: XYZ`` and ``Date: YYYY-MM-DD`` lines.

    Rules:
        - Ticker: case-insensitive match of ``Ticker: <1-6 uppercase letters>``.
        - Date: case-insensitive match of ``Date: YYYY-MM-DD``, parsed with
          ``datetime.strptime``.
        - Returns ``None`` for whichever field is not found.
    """
    ticker: Optional[str] = None
    date: Optional[datetime] = None

    m_ticker = re.search(r"(?i)^Ticker:\s*([A-Za-z]{1,6})", text, re.MULTILINE)
    if m_ticker:
        ticker = m_ticker.group(1).upper()
    else:
        # Fallback: look for "(AAPL)"-style ticker in the text
        m_paren = re.search(r"\(([A-Z]{1,6})\)", text)
        if m_paren:
            ticker = m_paren.group(1)

    m_date = re.search(r"(?i)^Date:\s*(\d{4}-\d{2}-\d{2})", text, re.MULTILINE)
    if m_date:
        try:
            date = datetime.strptime(m_date.group(1), "%Y-%m-%d")
        except ValueError:
            date = None

    return ticker, date

def _extract_ticker_from_filename(filepath: Path) -> Optional[str]:
    """
    Try to infer a ticker symbol from the filename.

    Looks for a 1-6 uppercase-letter token delimited by underscores, hyphens,
    or the start/end of the stem.  Returns the first match or ``None``.

    Examples:
        AAPL_news.csv          -> AAPL
        bloomberg_AAPL_prices  -> AAPL
    """
    stem = filepath.stem
    parts = re.split(r"[_\-\s]+", stem)
    for part in parts:
        if re.fullmatch(r"[A-Za-z]{1,6}", part):
            return part.upper()
    return None


# Backward-compatible alias so existing imports still work:
#   from document_loader import Document
Document = DocumentSchema


class DocumentLoader:
    """
    Simple loader for Bloomberg-exported documents.

    Supports: TXT, CSV
    Placeholder: PDF (not implemented)
    """

    def load_txt(
        self,
        filepath: Union[str, Path],
        ticker: Optional[str] = None,
        source: str = "bloomberg_export",
    ) -> list[DocumentSchema]:
        """
        Load a plain text document.

        Args:
            filepath: Path to the .txt file
            ticker: Optional ticker to attach to the document
            source: Source tag stored in the schema

        Returns:
            List containing one DocumentSchema object
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        extracted_ticker, extracted_date = _extract_ticker_and_date_from_text(content)
        final_ticker = ticker if ticker is not None else extracted_ticker

        doc = DocumentSchema(
            content=content,
            source=source,
            ticker=final_ticker,
            date=extracted_date,
            filepath=str(path),
        )
        return [doc]

    def load_csv(
        self,
        filepath: Union[str, Path],
        content_column: Optional[str] = None,
        ticker: Optional[str] = None,
        source: str = "bloomberg_export",
        date_column: Optional[str] = None,
        ticker_column: Optional[str] = None,
        max_rows: Optional[int] = None,
    ) -> list[DocumentSchema]:
        """
        Load a CSV file, optionally extracting content from a specific column.

        Args:
            filepath: Path to the .csv file.
            content_column: Column whose values become document content
                (one doc per row).  If *None*, the entire CSV is serialised
                as a single document.
            ticker: Explicit ticker applied to every document.  Falls back
                to *ticker_column* per row, then filename inference.
            source: Source tag stored in the schema.
            date_column: Column containing dates.  Parsed per row via
                ``pd.to_datetime(..., errors="coerce")``.
            ticker_column: Column containing per-row ticker symbols.
            max_rows: If set, only the first *max_rows* rows are processed.

        Returns:
            List of DocumentSchema objects.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        df = pd.read_csv(path)

        if df.empty:
            raise ValueError(f"CSV file is empty: {path}")

        # Resolve a file-level ticker (used when no per-row source exists)
        file_ticker = ticker if ticker is not None else _extract_ticker_from_filename(path)

        documents: list[DocumentSchema] = []

        if content_column:
            # ── row-per-document mode ─────────────────────────────────────
            if content_column not in df.columns:
                raise ValueError(
                    f"Column '{content_column}' not found in CSV. "
                    f"Available columns: {list(df.columns)}"
                )

            if max_rows is not None:
                df = df.head(max_rows)

            for _, row in df.iterrows():
                raw_content = row[content_column]
                # Skip empty / NaN content
                if pd.isna(raw_content) or str(raw_content).strip() == "":
                    continue

                # --- date ---
                row_date: Optional[datetime] = None
                if date_column and date_column in df.columns:
                    parsed = pd.to_datetime(row[date_column], errors="coerce")
                    if pd.notna(parsed):
                        row_date = parsed.to_pydatetime()

                # --- ticker ---
                row_ticker: Optional[str] = None
                if ticker is not None:
                    row_ticker = ticker
                elif ticker_column and ticker_column in df.columns:
                    val = row[ticker_column]
                    if pd.notna(val) and str(val).strip():
                        row_ticker = str(val).strip().upper()
                else:
                    row_ticker = file_ticker

                doc = DocumentSchema(
                    content=str(raw_content),
                    source=source,
                    ticker=row_ticker,
                    date=row_date,
                    filepath=str(path),
                )
                documents.append(doc)
        else:
            # ── whole-CSV fallback ────────────────────────────────────────
            csv_content = df.to_string()
            doc = DocumentSchema(
                content=csv_content,
                source=source,
                ticker=file_ticker,
                date=None,
                filepath=str(path),
            )
            documents.append(doc)

        return documents

    def chunk_documents(
        self,
        docs: list[DocumentSchema],
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ) -> list[DocumentSchema]:
        """Split each document's content into overlapping windows.

        Every chunk inherits the ticker, date, source, and filepath of its
        parent document so that retrieval metadata is preserved end-to-end.

        Short documents (content length ≤ chunk_size) are returned as a
        single-element list — no copying or truncation occurs.

        Args:
            docs: Documents to chunk.
            chunk_size: Maximum characters per window (default 800).
            overlap: Characters shared between adjacent windows (default 100).

        Returns:
            Flat list of DocumentSchema chunks (≥ len(docs) items).
        """
        result: list[DocumentSchema] = []
        for doc in docs:
            for window in chunk_text(doc.content, chunk_size, overlap):
                result.append(
                    DocumentSchema(
                        content=window,
                        source=doc.source,
                        ticker=doc.ticker,
                        date=doc.date,
                        filepath=doc.filepath,
                        about_score=doc.about_score,
                    )
                )
        return result

    def load_pdf(self, filepath: Union[str, Path]) -> list[DocumentSchema]:
        """
        Load a PDF document.

        Currently NOT implemented. PDF parsing requires external dependencies
        and OCR strategy (PyPDF, pdfplumber, or similar).

        Raises:
            NotImplementedError: Always, as PDF support is not yet implemented
        """
        raise NotImplementedError(
            "PDF loading not yet implemented. "
            "For now, export Bloomberg data as CSV or TXT. "
            "PDF parsing will be added in a future release."
        )


if __name__ == "__main__":
    """
    Simple demonstration of DocumentLoader with DocumentSchema output.
    """
    loader = DocumentLoader()

    # ── TXT (with extraction) ─────────────────────────────────────────────
    print("=" * 60)
    print("Example 1: Loading sample_document.txt (ticker/date extraction)")
    print("=" * 60)

    sample_txt_path = Path("sample_document.txt")
    if sample_txt_path.exists():
        docs_txt = loader.load_txt(sample_txt_path)  # no ticker override
        for doc in docs_txt:
            print(f"  type    : {type(doc).__name__}")
            print(f"  ticker  : {doc.ticker}")
            print(f"  date    : {doc.date}  (type={type(doc.date).__name__})")
            print(f"  content : {doc.content[:60]}...")
    else:
        print("  sample_document.txt not found – skipping.")

    # ── CSV (per-row with date + ticker columns) ─────────────────────────
    print("\n" + "=" * 60)
    print("Example 2: CSV per-row ingestion (date_column + ticker from filename)")
    print("=" * 60)

    sample_csv_rows = Path("AAPL_research.csv")
    if not sample_csv_rows.exists():
        pd.DataFrame(
            {
                "date": ["2025-07-24", "2025-07-25", "2025/07/28", None],
                "summary": [
                    "Apple beats Q3 earnings estimates",
                    "Supply chain resilience confirmed by analysts",
                    "New AI product line announced at WWDC",
                    "",  # empty row – should be skipped
                ],
            }
        ).to_csv(sample_csv_rows, index=False)

    docs_rows = loader.load_csv(
        sample_csv_rows,
        content_column="summary",
        date_column="date",
        max_rows=3,
    )
    print(f"  Loaded {len(docs_rows)} document(s) (empty rows skipped)")
    for i, doc in enumerate(docs_rows[:2], 1):
        date_info = f"{doc.date}  (type={type(doc.date).__name__})" if doc.date else "None"
        print(f"  [{i}] ticker={doc.ticker}  date={date_info}")
        print(f"       content={doc.content[:60]}")

    sample_csv_rows.unlink(missing_ok=True)

    # ── CSV (whole file fallback) ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Example 3: CSV whole-file fallback (ticker from filename)")
    print("=" * 60)

    sample_csv_whole = Path("MSFT_prices.csv")
    if not sample_csv_whole.exists():
        pd.DataFrame(
            {"Close": [410.50, 412.00], "Volume": [30_000_000, 28_000_000]}
        ).to_csv(sample_csv_whole, index=False)

    docs_whole = loader.load_csv(sample_csv_whole)
    for doc in docs_whole:
        print(f"  ticker  : {doc.ticker}")
        print(f"  content : {doc.content[:80]}...")

    sample_csv_whole.unlink(missing_ok=True)

    # ── PDF placeholder ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Example 4: Attempting to load PDF (expected to fail)")
    print("=" * 60)

    try:
        loader.load_pdf("dummy.pdf")
    except NotImplementedError as e:
        print(f"  Expected error: {e}")

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
