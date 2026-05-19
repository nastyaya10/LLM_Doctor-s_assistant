import argparse
from pathlib import Path

from parse_pdf import parse_pdf_to_json


DB_DIR = Path(__file__).resolve().parent
ASSETS_DIR = DB_DIR / "assets"
OUTPUT_DIR = DB_DIR / "output"
PDF_PATTERN = "*.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse all PDF files in db/assets.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild JSON files even when they already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf_files = sorted(ASSETS_DIR.glob(PDF_PATTERN))

    if not pdf_files:
        print(f"No PDF files found in: {ASSETS_DIR}")
        return 1

    failed_files = []

    for pdf_path in pdf_files:
        json_path = OUTPUT_DIR / f"{pdf_path.stem}.json"
        if json_path.exists() and not args.force:
            print(f"Already parsed, skip: {pdf_path.name}")
            continue

        print(f"Parsing: {pdf_path.name}")
        try:
            json_path = parse_pdf_to_json(pdf_path, OUTPUT_DIR)
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
