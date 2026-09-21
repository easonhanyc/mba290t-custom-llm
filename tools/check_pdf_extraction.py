"""Check what pypdf actually extracts from every PDF in corpus/.

The notebook imports PDFs through pypdf and warns about pages with no extractable
text, but it does not show whether the extracted words match the source. This
script does: it re-extracts each PDF page by page, reports page count, empty
pages, and character counts, and - when a plain-text twin of the PDF exists in
corpus/ - compares the two token streams so silent extraction damage is visible.

    python tools/check_pdf_extraction.py
"""
import io
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pypdf import PdfReader  # noqa: E402

from run_evals import load_suite, matching_cases, word_tokens  # noqa: E402

# Each corpus folder's PDF has its plain-text original in docs/, outside every
# training input, so the PDF is the only training source for that text while the
# extraction stays diffable against a known original.
CORPUS_DIRS = ["corpus", "corpus_seven", "corpus_unpaired"]


def twin_for(pdf_path):
    return ROOT / "docs" / f"pdf_source_{pdf_path.parent.name}.txt"


def main():
    suite = load_suite(ROOT / "evals/language_evals.json")
    pdfs = [q for d in CORPUS_DIRS for q in sorted((ROOT / d).rglob("*.pdf"))]
    if not pdfs:
        print("No PDFs in corpus/.")
        return 0
    report = []
    for path in pdfs:
        data = path.read_bytes()
        reader = PdfReader(io.BytesIO(data))
        pages, empty = [], []
        for number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            pages.append(text)
            if not text.strip():
                empty.append(number)
        extracted = "\n".join(pages)
        tokens = word_tokens(extracted)
        entry = {
            "file": str(path.relative_to(ROOT)),
            "bytes": len(data),
            "encrypted": reader.is_encrypted,
            "pages": len(reader.pages),
            "pages_with_no_text": empty,
            "characters_extracted": len(extracted),
            "tokens_extracted": len(tokens),
            "distinct_tokens": len(set(tokens)),
            "eval_prompts_found": matching_cases(extracted, suite),
        }
        twin = twin_for(path)
        if twin.exists():
            source = word_tokens(twin.read_text(encoding="utf-8"))
            entry["compared_with"] = str(twin.relative_to(ROOT))
            entry["source_tokens"] = len(source)
            entry["token_multisets_identical"] = Counter(source) == Counter(tokens)
            entry["token_sequence_identical"] = source == tokens
            only_pdf = sorted(set(tokens) - set(source))
            only_src = sorted(set(source) - set(tokens))
            entry["types_only_in_pdf"] = only_pdf
            entry["types_only_in_source"] = only_src
        report.append(entry)
        print(json.dumps(entry, indent=2))

    out = ROOT / "results" / "pdf_extraction_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nWrote", out.relative_to(ROOT))
    bad = [e for e in report if e["eval_prompts_found"]]
    if bad:
        raise SystemExit(f"Eval prompts found inside a PDF: {bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
