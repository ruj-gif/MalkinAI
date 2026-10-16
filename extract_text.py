"""Extract page text from all PDFs in the project's data folder."""

import json
from io import BytesIO
import os
import shutil
from pathlib import Path

try:
    import pymupdf
except ImportError as exc:
    raise SystemExit(
        "PyMuPDF is required. Install dependencies with: "
        "python -m pip install -r requirements.txt"
    ) from exc

try:
    import pytesseract
except ImportError:
    pytesseract = None


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_PATH = PROJECT_DIR / "processed" / "extracted_pages.json"
PREVIEW_LENGTH = 240
OCR_DPI = 300


def configure_ocr() -> str | None:
    """Configure pytesseract if its Python package and Tesseract are available."""
    if pytesseract is None:
        print(
            "OCR unavailable: install the Python wrapper with "
            "'python -m pip install -r requirements.txt'."
        )
        return None

    # TESSERACT_CMD can point to tesseract.exe when it is not on PATH.
    configured_path = os.environ.get("TESSERACT_CMD")
    executable = configured_path or shutil.which("tesseract")
    if not executable or (configured_path and not Path(configured_path).is_file()):
        print(
            "OCR unavailable: Tesseract OCR was not found. Install Tesseract for Windows, "
            "then add its folder to PATH or set TESSERACT_CMD to the full path of "
            "tesseract.exe. Text extraction will continue without OCR."
        )
        return None

    pytesseract.pytesseract.tesseract_cmd = executable
    try:
        version = pytesseract.get_tesseract_version()
    except Exception as exc:
        print(f"OCR unavailable: Tesseract could not be started ({exc}).")
        return None
    print(f"OCR available: Tesseract {version}")
    return executable


def extract_pdf(pdf_path: Path, ocr_available: bool) -> tuple[list[dict], int, int, list[str]]:
    """Extract text, using OCR only when a page has no normal text."""
    records = []
    warnings = []
    character_count = 0

    with pymupdf.open(pdf_path) as document:
        page_count = len(document)
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text")
            method = "text"
            ocr_status = "not_needed"
            if not text.strip():
                method = "none"
                ocr_status = "unavailable" if not ocr_available else "failed"
                if ocr_available:
                    print(f"  OCR: {pdf_path.name}, page {page_number}: processing...")
                    try:
                        import pytesseract

                        scale = OCR_DPI / 72
                        pixmap = page.get_pixmap(
                            matrix=pymupdf.Matrix(scale, scale), alpha=False
                        )
                        from PIL import Image

                        image = Image.open(BytesIO(pixmap.tobytes("png")))
                        ocr_text = pytesseract.image_to_string(image, lang="eng")
                        if ocr_text.strip():
                            text = ocr_text
                            method = "ocr"
                            ocr_status = "succeeded"
                            print(
                                f"  OCR: {pdf_path.name}, page {page_number}: "
                                f"succeeded ({len(text)} characters)."
                            )
                        else:
                            ocr_status = "no_text"
                            print(
                                f"  OCR: {pdf_path.name}, page {page_number}: "
                                "completed but returned no text; scanned page may need review."
                            )
                    except Exception as exc:
                        ocr_status = "failed"
                        print(
                            f"  OCR: {pdf_path.name}, page {page_number}: failed ({exc})."
                        )
                else:
                    warnings.append(
                        f"  Page {page_number}: no extractable text; OCR skipped because "
                        "Tesseract is unavailable. Scanned pages may require OCR."
                    )
            character_count += len(text)
            records.append(
                {
                    "source": pdf_path.name,
                    "page": page_number,
                    "text": text,
                    "extraction_method": method,
                    "ocr_status": ocr_status,
                }
            )

    return records, page_count, character_count, warnings


def main() -> None:
    if not DATA_DIR.is_dir():
        raise SystemExit(f"Data folder not found: {DATA_DIR}")

    pdf_files = sorted(
        (path for path in DATA_DIR.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: path.name.lower(),
    )
    if not pdf_files:
        raise SystemExit(f"No PDF files found in {DATA_DIR}")

    ocr_available = configure_ocr() is not None
    all_records = []
    for pdf_path in pdf_files:
        print(f"\nPDF: {pdf_path.name}")
        try:
            records, page_count, character_count, warnings = extract_pdf(
                pdf_path, ocr_available
            )
            all_records.extend(records)
            print(f"  Pages: {page_count}")
            print(f"  Characters extracted: {character_count}")
            preview = next((r["text"].strip() for r in records if r["text"].strip()), "")
            if preview:
                preview = " ".join(preview.split())
                if len(preview) > PREVIEW_LENGTH:
                    preview = preview[:PREVIEW_LENGTH].rstrip() + "..."
                print(f"  Preview: {preview}")
            else:
                print("  Preview: (no extractable text)")
            for warning in warnings:
                print(warning)
            ocr_successes = sum(1 for record in records if record["ocr_status"] == "succeeded")
            ocr_failures = sum(
                1 for record in records if record["ocr_status"] in {"failed", "no_text"}
            )
            if ocr_successes or ocr_failures:
                print(f"  OCR summary: {ocr_successes} succeeded, {ocr_failures} failed/no text")
        except Exception as exc:
            print(f"  ERROR: Could not read this PDF: {exc}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved {len(all_records)} page records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
