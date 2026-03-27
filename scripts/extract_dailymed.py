#!/usr/bin/env python3
import argparse
import os
import re
import zipfile
from pathlib import Path


def safe_name(s: str) -> str:
    # Keep filenames filesystem-friendly
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s.strip("_") or "file"


def extract_xmls_from_zip(zip_path: Path, dest_dir: Path, flatten: bool = True) -> int:
    xml_count = 0
    zip_stem = zip_path.stem

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            # Only files (not directories) ending with .xml
            if member.is_dir():
                continue
            name = member.filename
            if not name.lower().endswith(".xml"):
                continue

            # Basic path traversal protection
            if ".." in Path(name).parts:
                continue

            base = os.path.basename(name)
            base_safe = safe_name(base)

            if flatten:
                out_name = f"{safe_name(zip_stem)}__{base_safe}"
                out_path = dest_dir / out_name
            else:
                # Preserve internal folders under dest (still safe against ..)
                out_path = dest_dir / Path(*Path(name).parts).as_posix()
                out_path = dest_dir / out_path.relative_to(dest_dir)

            out_path.parent.mkdir(parents=True, exist_ok=True)

            with zf.open(member, "r") as src, open(out_path, "wb") as dst:
                dst.write(src.read())

            xml_count += 1

    return xml_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract only .xml files from DailyMed ZIPs into a flat folder."
    )
    parser.add_argument("--src", type=Path, required=True, help="Directory containing daily med .zip files")
    parser.add_argument("--dest", type=Path, required=True, help="Directory to write extracted .xml files")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of ZIPs processed")
    parser.add_argument("--flatten", action="store_true", help="Flatten output filenames (recommended)")
    parser.add_argument("--no-flatten", dest="flatten", action="store_false", help="Preserve zip internal paths")
    parser.set_defaults(flatten=True)

    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)

    zips = sorted([p for p in args.src.glob("*.zip") if p.is_file()])
    if args.limit is not None:
        zips = zips[: args.limit]

    total_xml = 0
    for i, zp in enumerate(zips, start=1):
        extracted = extract_xmls_from_zip(zp, args.dest, flatten=args.flatten)
        total_xml += extracted
        print(f"[{i}/{len(zips)}] {zp.name}: extracted {extracted} XML files (running total: {total_xml})")

    print(f"Done. Total XML extracted: {total_xml}")
    print(f"Output folder: {args.dest}")


if __name__ == "__main__":
    main()