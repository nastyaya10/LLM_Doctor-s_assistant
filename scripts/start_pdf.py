import argparse
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path, чтобы импортировать пути
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.services.paths import RAW_DATA, PARSED_DATA
from scripts.pdf_parser import parse_pdf_to_json  # Убедись, что pdf_parser теперь в scripts/

PDF_PATTERN = "*.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse all PDF files in data/raw.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild JSON files even when they already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Теперь используем новые пути
    input_dir = RAW_DATA
    output_dir = PARSED_DATA

    pdf_files = sorted(input_dir.glob(PDF_PATTERN))

    if not pdf_files:
        print(f"No PDF files found in: {input_dir}")
        return 1

    failed_files = []

    for pdf_path in pdf_files:
        json_path = output_dir / f"{pdf_path.stem}.json"
        if json_path.exists() and not args.force:
            print(f"Already parsed, skip: {pdf_path.name}")
            continue

        print(f"Parsing: {pdf_path.name}")
        try:
            parse_pdf_to_json(pdf_path, output_dir)
        except Exception as error:
            failed_files.append((pdf_path, error))
            print(f"  Failed: {error}")
            continue

        print(f"  JSON saved to: {json_path}")

    if failed_files:
        print("\nSome files failed:")
        for pdf_path, error in failed_files:
            print(f"- {pdf_path.name}: {error}")
        return 1

    print("\nAll PDF files parsed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
