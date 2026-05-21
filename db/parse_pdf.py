import argparse
import html
import json
import os
import re
import ssl
import sys
import tempfile
from pathlib import Path
from typing import Any, NamedTuple


PARSER_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PARSER_DIR.parent
DEFAULT_INPUT = PARSER_DIR / "assets" / "1596_1582.pdf"
DEFAULT_OUTPUT_DIR = PARSER_DIR / "output"
LOCAL_CACHE_DIR = PROJECT_DIR / ".cache"
DEFAULT_OCR_DPI = int(os.getenv("PARSER_OCR_DPI", "200"))
USE_EASYOCR_FALLBACK = os.getenv("PARSER_USE_EASYOCR", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MAX_EASYOCR_PAGES = int(os.getenv("PARSER_EASYOCR_MAX_PAGES", "6"))


def configure_ssl_certificates() -> None:
    try:
        import certifi
    except ImportError:
        return

    cafile = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", cafile)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", cafile)
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=cafile)


configure_ssl_certificates()
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(LOCAL_CACHE_DIR / "paddlex"))
os.environ.setdefault("PADDLE_HOME", str(LOCAL_CACHE_DIR / "paddle"))
os.environ.setdefault("HF_HOME", str(LOCAL_CACHE_DIR / "huggingface"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse PDF documents with Docling and export compact JSON."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to a PDF file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for parsed JSON output.",
    )
    return parser.parse_args()


def load_docling_converter():
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as error:
        raise SystemExit(
            "Docling is not installed. Install it with:\n"
            "  python -m pip install -r requirements.txt"
        ) from error

    return DocumentConverter


class TextCandidate(NamedTuple):
    source: str
    text: str
    score: float


CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
OCR_GARBAGE_RE = re.compile(
    r"(?i)(?:"
    r"др\s*ьртамент|здав[о0]{2}\s*лнения|щентраль|"
    r"организач|медпц|заключеп|состоянип|паправ|прохожденпе|"
    r"утверл|методп|клипи|рекоменлач|обшероссий|россп|"
    r"пр€|€|ý|[)(]{2,}"
    r")"
)

LATIN_TO_CYRILLIC = str.maketrans(
    {
        "A": "А",
        "B": "В",
        "C": "С",
        "E": "Е",
        "H": "Н",
        "K": "К",
        "M": "М",
        "O": "О",
        "P": "Р",
        "T": "Т",
        "X": "Х",
        "Y": "У",
        "a": "а",
        "c": "с",
        "e": "е",
        "o": "о",
        "p": "р",
        "x": "х",
        "y": "у",
    }
)

NUMERIC_OCR_REPLACEMENTS = str.maketrans(
    {
        "O": "0",
        "О": "0",
        "o": "0",
        "о": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "З": "3",
        "з": "3",
        "S": "5",
        "s": "5",
        "Б": "6",
    }
)


def normalize_text(text: str) -> str:
    text = re.sub(r"<!--\s*image\s*-->", " ", text, flags=re.IGNORECASE)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_ocr_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<!--\s*image\s*-->", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"([№N])\s*[_\-]?\s*(\d)", r"№ \2", text)

    # Fix latin homoglyphs only inside words that are mostly Cyrillic.
    def fix_mixed_cyrillic_word(match: re.Match[str]) -> str:
        word = match.group(0)
        cyrillic_count = len(CYRILLIC_RE.findall(word))
        if cyrillic_count >= max(1, len(word) // 3):
            return word.translate(LATIN_TO_CYRILLIC)
        return word

    text = re.sub(r"[A-Za-zА-Яа-яЁё]{3,}", fix_mixed_cyrillic_word, text)

    # Fix common OCR mistakes in numeric fragments without touching normal words.
    def fix_numeric_fragment(match: re.Match[str]) -> str:
        fragment = match.group(0)
        return re.sub(
            r"\S+",
            lambda token_match: (
                token_match.group(0)
                if re.match(r"[A-Za-zА-Яа-яЁё]\d", token_match.group(0))
                else token_match.group(0).translate(NUMERIC_OCR_REPLACEMENTS)
            ),
            fragment,
        )

    text = re.sub(
        r"(?<![A-Za-zА-Яа-яЁё])[\dOoОоIl|ЗзSsБ][\dOoОоIl|ЗзSsБ\s.,:/\\\-]{1,}"
        r"(?=[\dOoОоIl|ЗзSsБ]|$)",
        fix_numeric_fragment,
        text,
    )
    text = re.sub(r"(?<=\d)\s+([.,:/\\-])\s+(?=\d)", r"\1", text)
    return normalize_text(text)


def text_quality_score(text: str) -> float:
    normalized = normalize_text(text)
    if not normalized:
        return 0.0

    alpha_chars = [char for char in normalized if char.isalpha()]
    cyrillic_count = len(CYRILLIC_RE.findall(normalized))
    alpha_count = len(alpha_chars) or 1
    cyrillic_ratio = cyrillic_count / alpha_count

    replacement_penalty = normalized.count("�") * 50
    mojibake_penalty = len(re.findall(r"[ÐÑ][\x80-\xbf]?", normalized)) * 30
    ocr_garbage_penalty = len(OCR_GARBAGE_RE.findall(normalized)) * 250
    length_score = min(len(normalized), 50_000) / 50

    return (
        length_score
        + (cyrillic_ratio * 1_000)
        - replacement_penalty
        - mojibake_penalty
        - ocr_garbage_penalty
    )


def extract_text_with_pypdf(input_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""

    reader = PdfReader(str(input_path))
    page_texts = [page.extract_text() or "" for page in reader.pages]
    return normalize_ocr_text("\n".join(page_texts))


def extract_text_with_pymupdf(input_path: Path) -> str:
    try:
        import fitz
    except ImportError:
        return ""

    page_texts = []
    with fitz.open(input_path) as document:
        for page in document:
            page_texts.append(page.get_text("text") or "")

    return normalize_ocr_text("\n".join(page_texts))


def extract_text_with_pdfplumber(input_path: Path) -> str:
    try:
        import pdfplumber
    except ImportError:
        return ""

    page_texts = []
    with pdfplumber.open(str(input_path)) as pdf:
        for page in pdf.pages:
            page_texts.append(
                page.extract_text(x_tolerance=1, y_tolerance=3, layout=False) or ""
            )

    return normalize_ocr_text("\n".join(page_texts))


def extract_tables_with_pdfplumber(input_path: Path) -> list[dict[str, Any]]:
    try:
        import pdfplumber
    except ImportError:
        return []

    tables = []
    with pdfplumber.open(str(input_path)) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            for table_index, rows in enumerate(page.extract_tables() or [], start=1):
                cleaned_rows = [
                    [normalize_text(str(cell or "")) for cell in row]
                    for row in rows
                    if row
                ]
                if cleaned_rows:
                    tables.append(
                        {
                            "table_id": f"page_{page_index}_table_{table_index}",
                            "caption": f"Таблица {len(tables) + 1}",
                            "data": cleaned_rows,
                        }
                    )

    return tables


def render_pdf_pages(input_path: Path, output_dir: Path, dpi: int = DEFAULT_OCR_DPI) -> list[Path]:
    try:
        import fitz
    except ImportError:
        return []

    image_paths = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(input_path) as document:
        for page_index, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = output_dir / f"page_{page_index:04d}.png"
            pixmap.save(image_path)
            image_paths.append(image_path)

    return image_paths


def line_sort_key(line: Any) -> tuple[float, float]:
    box = line[0] if isinstance(line, (list, tuple)) and line else []
    if not isinstance(box, (list, tuple)) or not box:
        return (0.0, 0.0)

    xs = []
    ys = []
    for point in box:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            xs.append(float(point[0]))
            ys.append(float(point[1]))

    if not xs or not ys:
        return (0.0, 0.0)

    return (min(ys), min(xs))


def extract_paddle_text_from_result(result: Any) -> str:
    if not result:
        return ""

    try:
        texts = result["rec_texts"]
    except (KeyError, TypeError):
        texts = None

    if isinstance(texts, list):
        return normalize_ocr_text(" ".join(str(text) for text in texts if str(text).strip()))

    if hasattr(result, "json"):
        json_value = result.json() if callable(result.json) else result.json
        return extract_paddle_text_from_result(json_value)

    if hasattr(result, "to_json"):
        json_value = result.to_json() if callable(result.to_json) else result.to_json
        return extract_paddle_text_from_result(json_value)

    if isinstance(result, dict):
        texts = result.get("rec_texts") or result.get("texts") or []
        if isinstance(texts, list):
            return normalize_ocr_text(" ".join(str(text) for text in texts))
        return ""

    if isinstance(result, list) and result and isinstance(result[0], dict):
        return normalize_ocr_text(
            " ".join(extract_paddle_text_from_result(item) for item in result)
        )

    pages = result if isinstance(result, list) else [result]
    page_texts = []
    for page in pages:
        if not page:
            continue
        lines = page
        if (
            isinstance(page, list)
            and len(page) == 1
            and isinstance(page[0], list)
            and page[0]
            and isinstance(page[0][0], (list, tuple))
        ):
            lines = page[0]

        line_texts = []
        for line in sorted(lines, key=line_sort_key):
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue
            payload = line[1]
            if isinstance(payload, (list, tuple)) and payload:
                line_texts.append(str(payload[0]))
            elif isinstance(payload, str):
                line_texts.append(payload)
        page_texts.append(" ".join(line_texts))

    return normalize_ocr_text("\n".join(page_texts))


def create_paddle_ocr() -> Any:
    from paddleocr import PaddleOCR

    option_sets = [
        {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": True,
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": "eslav_PP-OCRv5_mobile_rec",
            "text_det_limit_side_len": 1280,
        },
        {
            "lang": "ru",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": True,
            "text_det_limit_side_len": 1280,
        },
        {"lang": "ru"},
    ]
    last_error = None

    for options in option_sets:
        try:
            return PaddleOCR(**options)
        except Exception as error:
            last_error = error

    if last_error:
        raise last_error
    raise RuntimeError("Failed to initialize PaddleOCR")


def extract_text_with_paddleocr(input_path: Path) -> str:
    try:
        ocr = create_paddle_ocr()
    except Exception as error:
        print(f"PaddleOCR is unavailable: {error}", file=sys.stderr)
        return ""

    page_texts = []
    with tempfile.TemporaryDirectory(prefix="paddleocr_pages_") as temp_dir:
        image_paths = render_pdf_pages(input_path, Path(temp_dir))
        for image_path in image_paths:
            try:
                if hasattr(ocr, "predict"):
                    result = ocr.predict(str(image_path))
                elif hasattr(ocr, "ocr"):
                    try:
                        result = ocr.ocr(str(image_path), cls=True)
                    except TypeError:
                        result = ocr.ocr(str(image_path))
                else:
                    return ""
            except Exception as error:
                print(f"PaddleOCR failed on {image_path.name}: {error}", file=sys.stderr)
                return ""
            page_texts.append(extract_paddle_text_from_result(result))

    return normalize_ocr_text("\n".join(page_texts))


def extract_text_with_easyocr(input_path: Path) -> str:
    if not USE_EASYOCR_FALLBACK:
        return ""

    try:
        import easyocr
    except ImportError:
        return ""

    try:
        reader = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)
    except Exception as error:
        print(f"EasyOCR is unavailable: {error}", file=sys.stderr)
        return ""

    page_texts = []
    with tempfile.TemporaryDirectory(prefix="easyocr_pages_") as temp_dir:
        image_paths = render_pdf_pages(input_path, Path(temp_dir))
        if MAX_EASYOCR_PAGES > 0:
            image_paths = image_paths[:MAX_EASYOCR_PAGES]

        for image_path in image_paths:
            try:
                result = reader.readtext(
                    str(image_path),
                    detail=0,
                    paragraph=False,
                    decoder="greedy",
                    batch_size=4,
                )
            except Exception as error:
                print(f"EasyOCR failed on {image_path.name}: {error}", file=sys.stderr)
                return ""
            page_texts.append(" ".join(str(text) for text in result if str(text).strip()))

    return normalize_ocr_text("\n".join(page_texts))


def choose_best_text(candidates: list[tuple[str, str]]) -> TextCandidate:
    scored_candidates = [
        TextCandidate(source, normalize_ocr_text(text), text_quality_score(text))
        for source, text in candidates
        if normalize_text(text)
    ]

    if not scored_candidates:
        return TextCandidate("empty", "", 0.0)

    return max(scored_candidates, key=lambda candidate: candidate.score)


def has_ocr_garbage(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return True

    garbage_count = len(OCR_GARBAGE_RE.findall(normalized))
    return garbage_count >= 3 or text_quality_score(normalized) < 900


def build_docling_converter(use_ocr: bool) -> tuple[Any, bool]:
    DocumentConverter = load_docling_converter()

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            EasyOcrOptions,
            PdfPipelineOptions,
            TableStructureOptions,
        )
        from docling.document_converter import PdfFormatOption
    except (ImportError, TypeError, ValueError):
        return DocumentConverter(), False

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = use_ocr
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableStructureOptions(
        do_cell_matching=True
    )
    if use_ocr:
        pipeline_options.ocr_options = EasyOcrOptions(
            lang=["ru", "en"],
            force_full_page_ocr=True,
        )

    return (
        DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        ),
        True,
    )


def convert_with_docling(input_path: Path, use_ocr: bool) -> Any:
    DocumentConverter = load_docling_converter()
    converter, has_custom_options = build_docling_converter(use_ocr)

    try:
        return converter.convert(input_path)
    except Exception as error:
        if not has_custom_options:
            raise
        print(
            f"Docling configured converter failed, retrying default converter: {error}",
            file=sys.stderr,
        )
        return DocumentConverter().convert(input_path)


def document_to_dict(document: Any) -> dict[str, Any]:
    if hasattr(document, "export_to_dict"):
        return document.export_to_dict()

    if hasattr(document, "model_dump"):
        return document.model_dump(mode="json")

    if hasattr(document, "dict"):
        return document.dict()

    return {}


def export_markdown(document: Any) -> str:
    if hasattr(document, "export_to_markdown"):
        return document.export_to_markdown()
    return ""


def extract_title(markdown: str, fallback: str) -> str:
    for raw_line in markdown.splitlines():
        line = normalize_text(raw_line)
        if not line:
            continue
        if line.startswith("#"):
            return line.lstrip("#").strip()
        return line

    return fallback


def extract_title_from_text(text: str, fallback: str) -> str:
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        line = normalize_text(sentence)
        if len(line) >= 8 and len(CYRILLIC_RE.findall(line)) >= 4:
            return line[:180]
    return fallback


def extract_text(markdown: str) -> str:
    lines = []
    in_table = False

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            in_table = False
            continue

        is_table_line = line.startswith("|") and line.endswith("|")
        if is_table_line:
            in_table = True
            continue

        if in_table and re.match(r"^[:\-\s|]+$", line):
            continue

        if line.startswith("#"):
            line = line.lstrip("#").strip()

        lines.append(line)
        in_table = False

    return normalize_text(" ".join(lines))


def get_caption(table_item: Any, table_index: int) -> str:
    for attr in ("caption", "label", "name"):
        value = getattr(table_item, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return f"Таблица {table_index}"


def table_data_to_rows(table_data: Any) -> list[list[str]]:
    if table_data is None:
        return []

    if hasattr(table_data, "export_to_dataframe"):
        dataframe = table_data.export_to_dataframe()
        return [
            [str(cell) for cell in row]
            for row in dataframe.fillna("").values.tolist()
        ]

    if hasattr(table_data, "model_dump"):
        table_data = table_data.model_dump(mode="json")
    elif hasattr(table_data, "dict"):
        table_data = table_data.dict()

    if isinstance(table_data, dict):
        grid = table_data.get("grid") or table_data.get("data") or []
        rows = []
        for row in grid:
            if isinstance(row, list):
                rows.append([cell_to_text(cell) for cell in row])
        return rows

    return []


def cell_to_text(cell: Any) -> str:
    if isinstance(cell, str):
        return normalize_text(cell)

    if isinstance(cell, dict):
        for key in ("text", "content", "value"):
            value = cell.get(key)
            if value is not None:
                return normalize_text(str(value))

    return normalize_text(str(cell))


def extract_tables_from_document(document: Any) -> list[dict[str, Any]]:
    tables = []
    table_items = getattr(document, "tables", []) or []

    for index, table_item in enumerate(table_items, start=1):
        table_data = getattr(table_item, "data", None)
        rows = table_data_to_rows(table_data)
        tables.append(
            {
                "table_id": f"table_{index}",
                "caption": get_caption(table_item, index),
                "data": rows,
            }
        )

    return tables


def extract_tables_from_dict(document_dict: dict[str, Any]) -> list[dict[str, Any]]:
    if not document_dict:
        return []

    raw_tables = document_dict.get("tables") or []
    tables = []

    for index, table in enumerate(raw_tables, start=1):
        caption = table.get("caption") or table.get("label") or f"Таблица {index}"
        raw_data = table.get("data") or {}
        rows = table_data_to_rows(raw_data)
        tables.append(
            {
                "table_id": f"table_{index}",
                "caption": normalize_text(str(caption)),
                "data": rows,
            }
        )

    return tables


def build_compact_json(
    document: Any,
    input_path: Path,
    text_candidate: TextCandidate,
) -> dict[str, Any]:
    markdown = export_markdown(document)
    document_dict = document_to_dict(document)
    docling_text = extract_text(markdown)
    best_text = choose_best_text(
        [
            (text_candidate.source, text_candidate.text),
            ("docling_markdown", docling_text),
        ]
    )

    tables = extract_tables_from_document(document)
    if not tables:
        tables = extract_tables_from_dict(document_dict)

    return {
        "title": extract_title(markdown, input_path.stem),
        "text": best_text.text,
        "tables": tables,
        "metadata": {
            "text_source": best_text.source,
            "text_quality_score": round(best_text.score, 2),
        },
    }


def build_text_only_json(
    input_path: Path,
    text_candidate: TextCandidate,
    error: Exception,
) -> dict[str, Any]:
    return {
        "title": input_path.stem,
        "text": text_candidate.text,
        "tables": [],
        "metadata": {
            "text_source": text_candidate.source,
            "text_quality_score": round(text_candidate.score, 2),
            "docling_error": str(error),
        },
    }


def build_text_layer_json(
    input_path: Path,
    text_candidate: TextCandidate,
    tables: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "title": extract_title_from_text(text_candidate.text, input_path.stem),
        "text": text_candidate.text,
        "tables": tables,
        "metadata": {
            "parser": "text_layer",
            "text_source": text_candidate.source,
            "text_quality_score": round(text_candidate.score, 2),
            "docling_ocr_enabled": False,
        },
    }


def build_paddle_ocr_json(
    input_path: Path,
    text_candidate: TextCandidate,
    tables: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "title": extract_title_from_text(text_candidate.text, input_path.stem),
        "text": text_candidate.text,
        "tables": tables,
        "metadata": {
            "parser": "paddleocr",
            "text_source": text_candidate.source,
            "text_quality_score": round(text_candidate.score, 2),
            "docling_ocr_enabled": False,
        },
    }


def parse_pdf_to_json(input_path: Path, output_dir: Path) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"PDF file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{input_path.stem}.json"

    pdf_text_candidate = choose_best_text(
        [
            ("pymupdf", extract_text_with_pymupdf(input_path)),
            ("pdfplumber", extract_text_with_pdfplumber(input_path)),
            ("pypdf", extract_text_with_pypdf(input_path)),
        ]
    )
    pdf_tables = extract_tables_with_pdfplumber(input_path)

    if pdf_text_candidate.text and not has_ocr_garbage(pdf_text_candidate.text):
        json_data = build_text_layer_json(input_path, pdf_text_candidate, pdf_tables)
        json_path.write_text(
            json.dumps(json_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return json_path

    paddle_text = extract_text_with_paddleocr(input_path)
    paddle_text_candidate = TextCandidate(
        "paddleocr",
        normalize_ocr_text(paddle_text),
        text_quality_score(paddle_text),
    )

    if paddle_text_candidate.text:
        best_ocr_candidate = choose_best_text(
            [
                (pdf_text_candidate.source, pdf_text_candidate.text),
                (paddle_text_candidate.source, paddle_text_candidate.text),
            ]
        )
        if (
            best_ocr_candidate.source == "paddleocr"
            and not has_ocr_garbage(best_ocr_candidate.text)
        ) or not pdf_text_candidate.text:
            json_data = build_paddle_ocr_json(input_path, best_ocr_candidate, pdf_tables)
            json_path.write_text(
                json.dumps(json_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return json_path

    easyocr_text = extract_text_with_easyocr(input_path)
    easyocr_text_candidate = TextCandidate(
        "easyocr",
        normalize_ocr_text(easyocr_text),
        text_quality_score(easyocr_text),
    )

    if easyocr_text_candidate.text:
        best_easyocr_candidate = choose_best_text(
            [
                (pdf_text_candidate.source, pdf_text_candidate.text),
                (paddle_text_candidate.source, paddle_text_candidate.text),
                (easyocr_text_candidate.source, easyocr_text_candidate.text),
            ]
        )
        if best_easyocr_candidate.source == "easyocr" or not pdf_text_candidate.text:
            json_data = build_paddle_ocr_json(input_path, best_easyocr_candidate, pdf_tables)
            json_data["metadata"]["parser"] = "easyocr"
            json_path.write_text(
                json.dumps(json_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return json_path

    try:
        result = convert_with_docling(input_path, use_ocr=True)
        json_data = build_compact_json(result.document, input_path, pdf_text_candidate)
        json_data["metadata"]["parser"] = "docling_ocr"
        json_data["metadata"]["docling_ocr_enabled"] = True
    except Exception as error:
        if not pdf_text_candidate.text:
            raise
        print(
            f"Docling failed, saving text-only JSON from {pdf_text_candidate.source}: {error}",
            file=sys.stderr,
        )
        json_data = build_text_only_json(input_path, pdf_text_candidate, error)

    json_path.write_text(
        json.dumps(json_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return json_path


def main() -> int:
    args = parse_args()

    try:
        json_path = parse_pdf_to_json(args.input, args.output_dir)
    except Exception as error:
        print(f"Failed to parse PDF: {error}", file=sys.stderr)
        return 1

    print(f"JSON saved to: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
