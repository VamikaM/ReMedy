"""Run and package the complete ReMedy Alzheimer V1 data-building workflow.

This is a reproducible data-build entry point. It produces a portable snapshot
ZIP containing curated tables and quality reports; raw API caches are omitted
because they can be regenerated from the cited public sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-source-refresh",
        action="store_true",
        help="Reuse existing ChEMBL and Open Targets files instead of rebuilding them.",
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=root / "exports" / "remedy_alzheimers_v1_snapshot.zip",
    )
    return parser.parse_args()


def run(root: Path, relative_script: str) -> None:
    print(f"\n>>> {relative_script}", flush=True)
    subprocess.run([sys.executable, relative_script], cwd=root, check=True)


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_tree(archive: ZipFile, path: Path, arc_root: Path, files: list[dict]) -> None:
    for file_path in sorted(path.rglob("*")):
        if not file_path.is_file():
            continue
        archive_name = str(arc_root / file_path.relative_to(path))
        archive.write(file_path, archive_name)
        files.append(
            {
                "archive_path": archive_name.replace("\\", "/"),
                "bytes": file_path.stat().st_size,
                "sha256": checksum(file_path),
            }
        )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]

    if not args.skip_source_refresh:
        run(root, "src/preprocessing/fetch_chembl_mechanisms.py")
        run(root, "src/preprocessing/fetch_open_targets_direction_evidence.py")
    run(root, "src/alzheimers/build_alzheimers_v1_dataset.py")
    run(root, "src/alzheimers/audit_alzheimers_v1_dataset.py")
    run(root, "src/alzheimers/expand_approved_drug_target_catalog.py")
    run(root, "src/alzheimers/fetch_alzheimers_target_evidence.py")

    curated = root / "data" / "curated" / "alzheimers_v1"
    audit = root / "reports" / "data_quality" / "alzheimers_v1"
    if not curated.exists():
        raise RuntimeError("Curated Alzheimer V1 directory was not produced.")
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)

    files: list[dict] = []
    with ZipFile(args.output_zip, "w", compression=ZIP_DEFLATED) as archive:
        add_tree(archive, curated, Path("remedy_alzheimers_v1") / "curated", files)
        if audit.exists():
            add_tree(archive, audit, Path("remedy_alzheimers_v1") / "quality_audit", files)
        manifest = {
            "artifact": "ReMedy Alzheimer V1 curated dataset snapshot",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "files": files,
            "rebuild_entrypoint": "src/alzheimers/build_and_package_alzheimers_v1.py",
            "source_policy": (
                "Tables retain their source provenance. Raw public API response caches are "
                "regenerable and intentionally excluded from the portable snapshot."
            ),
        }
        archive.writestr(
            "remedy_alzheimers_v1/snapshot_manifest.json",
            json.dumps(manifest, indent=2),
        )

    print(json.dumps(
        {
            "snapshot": str(args.output_zip),
            "bytes": args.output_zip.stat().st_size,
            "included_files": len(files),
        },
        indent=2,
    ))
    print("Download this ZIP or upload it as a private Kaggle Dataset before model work.")


if __name__ == "__main__":
    main()
