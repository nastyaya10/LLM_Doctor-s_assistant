import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


PARSER_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PARSER_DIR / "assets" / "Клинические_рекомендации.pdf"
DEFAULT_OUTPUT_DIR = PARSER_DIR / "output"


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
            "  python -m pip install -r pdf_parser/requirements.txt"
        ) from error

    return DocumentConverter


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


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
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            return line.lstrip("#").strip()
        return line

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


def build_compact_json(document: Any, input_path: Path) -> dict[str, Any]:
    markdown = export_markdown(document)
    document_dict = document_to_dict(document)

    tables = extract_tables_from_document(document)
    if not tables:
        tables = extract_tables_from_dict(document_dict)

    return {
        "title": extract_title(markdown, input_path.stem),
        "text": extract_text(markdown),
        "tables": tables,
    }


def parse_pdf_to_json(input_path: Path, output_dir: Path) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"PDF file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{input_path.stem}.json"

    DocumentConverter = load_docling_converter()
    converter = DocumentConverter()

    result = converter.convert(input_path)
    json_data = build_compact_json(result.document, input_path)

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
