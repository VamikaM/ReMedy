"""Package an existing ReMedy Alzheimer V1 curated dataset as a portable ZIP."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def checksum(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def add_tree(archive: ZipFile, source: Path, archive_root: Path, files: list[dict]) -> None:
    for file_path in sorted(source.rglob("*")):
        if file_path.is_file():
            name = str(archive_root / file_path.relative_to(source)).replace("\\", "/")
            archive.write(file_path, name)
            files.append({"archive_path": name, "bytes": file_path.stat().st_size, "sha256": checksum(file_path)})


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=root / "exports" / "remedy_alzheimers_v1_snapshot_updated.zip",
    )
    args = parser.parse_args()

    curated = root / "data" / "curated" / "alzheimers_v1"
    audit = root / "reports" / "data_quality" / "alzheimers_v1"
    if not (curated / "manifest.json").exists():
        raise FileNotFoundError(f"Missing curated dataset manifest: {curated / 'manifest.json'}")

    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    files: list[dict] = []
    with ZipFile(args.output_zip, "w", compression=ZIP_DEFLATED) as archive:
        add_tree(archive, curated, Path("remedy_alzheimers_v1") / "curated", files)
        if audit.exists():
            add_tree(archive, audit, Path("remedy_alzheimers_v1") / "quality_audit", files)
        archive.writestr(
            "remedy_alzheimers_v1/snapshot_manifest.json",
            json.dumps(
                {
                    "artifact": "ReMedy Alzheimer V1 curated dataset snapshot",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "included_files": files,
                    "source_policy": "Portable tables retain source provenance; raw API caches are regenerable.",
                },
                indent=2,
            ),
        )
    print(json.dumps({"snapshot": str(args.output_zip), "included_files": len(files), "bytes": args.output_zip.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
