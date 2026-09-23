#!/usr/bin/env python3
"""
audit_release.py — pre-submission checks on the artefact that ships.

Two reviewers rated the submission "Datasets: 1 = No usable datasets submitted"
despite a data attachment, so the release has to be self-evidently complete and
self-evidently anonymous. This runs both checks and writes a manifest.

    python3 tools/audit_release.py                 # anonymity + manifest
    python3 tools/audit_release.py --strict        # non-zero exit on any finding

Checks:
  1. anonymity  — author names, the institution, absolute Windows/Unix paths,
     e-mail addresses, and anything shaped like an API key;
  2. secrets    — .env files, key-like literals in tracked source;
  3. manifest   — SHA-256 for every released file, so a reviewer can verify the
     exact artefact behind a number;
  4. regeneration — checks that the tables and figures the paper needs exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("platos_ship3.audit_release")

SKIP_DIRS = {".git", "__pycache__", "_shards", "node_modules", ".venv", "venv",
             ".ipynb_checkpoints"}

IDENTITY_PATTERNS = [
    (r"\bKoushik\b", "author given name"),
    (r"\bDeb\b(?!ug|ian)", "author surname"),
    (r"\bAbhinaba\b", "co-author given name"),
    (r"\bBasu\b", "co-author surname"),
    (r"IIIT[\s-]?Kalyani", "institution"),
    (r"\bDevDaring\b", "personal GitHub handle"),
    (r"[\w.+-]+@[\w-]+\.[\w.]+", "e-mail address"),
    (r"[A-Za-z]:\\\\?Users\\\\?[^\\\\/\s\"']+", "absolute Windows user path"),
    (r"/home/[^/\s\"']+/", "absolute Unix home path"),
]

SECRET_PATTERNS = [
    (r"sk-[A-Za-z0-9]{16,}", "OpenAI-style key"),
    (r"sk-or-v1-[A-Za-z0-9]{16,}", "OpenRouter key"),
    (r"gh[pous]_[A-Za-z0-9]{20,}", "GitHub token"),
    (r"AIza[0-9A-Za-z_-]{20,}", "Google API key"),
    (r"(?i)api[_-]?key\s*[=:]\s*[\"'][^\"'\s]{12,}[\"']", "inline API key"),
]

TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".txt", ".tex", ".json",
                 ".sh", ".cfg", ".toml", ".ini", ".csv"}

# Produced by any analysis run, whichever protocol is present.
REQUIRED_OUTPUTS = [
    "results/tables/tab_main.tex",
    "results/tables/tab_contrasts.tex",
    "results/tables/tab_gradient.tex",
    "results/figures/figure1_capability_gradient.png",
    "results/outputs/registry.parquet",
    "results/outputs/paper_numbers.json",
]

# Required only once Protocol-B (Phase 3) runs exist; before that the legacy
# variants are the correct output and demanding these would be a false alarm.
REQUIRED_OUTPUTS_PROTOCOL_B = [
    "results/tables/tab_coverage.tex",
    "results/tables/tab_decomposition.tex",
    "results/outputs/r0_cache.parquet",
    "results/outputs/revision_log.parquet",
]


def _iter_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def check_anonymity(root: Path) -> List[Dict[str, Any]]:
    findings = []
    # This file necessarily contains the patterns it searches for, and so does
    # the report it writes; scanning either would report its own findings back.
    self_path = Path(__file__).resolve()
    report_path = (root / "results" / "outputs" / "release_audit.json").resolve()
    for path in _iter_files(root):
        if path.resolve() in (self_path, report_path):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern, description in IDENTITY_PATTERNS:
            for match in re.finditer(pattern, text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append({
                    "kind": "identity",
                    "file": str(path.relative_to(root)),
                    "line": line,
                    "what": description,
                    "match": match.group(0)[:80],
                })
    return findings


def check_secrets(root: Path) -> List[Dict[str, Any]]:
    findings = []
    self_path = Path(__file__).resolve()
    report_path = (root / "results" / "outputs" / "release_audit.json").resolve()
    for path in _iter_files(root):
        if path.resolve() in (self_path, report_path):
            continue
        if path.name == ".env" or path.name.startswith(".env."):
            if path.name != ".env.example":
                findings.append({"kind": "secret", "file": str(path.relative_to(root)),
                                 "line": 0, "what": "environment file present",
                                 "match": path.name})
                continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern, description in SECRET_PATTERNS:
            for match in re.finditer(pattern, text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append({
                    "kind": "secret",
                    "file": str(path.relative_to(root)),
                    "line": line,
                    "what": description,
                    "match": match.group(0)[:12] + "…",
                })
    return findings


def build_manifest(root: Path) -> Dict[str, Any]:
    entries = {}
    for path in _iter_files(root):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        entries[str(path.relative_to(root)).replace("\\", "/")] = {
            "sha256": digest.hexdigest(),
            "bytes": path.stat().st_size,
        }
    return entries


def check_required_outputs(root: Path) -> List[Dict[str, Any]]:
    findings = []
    required = list(REQUIRED_OUTPUTS)
    # Only demand the Protocol-B outputs once a Phase-3 run has happened.
    if (root / "results" / "outputs" / "revision_log.parquet").exists():
        required += REQUIRED_OUTPUTS_PROTOCOL_B
    for relative in required:
        if not (root / relative).exists():
            findings.append({
                "kind": "missing_output",
                "file": relative,
                "line": 0,
                "what": "required by the paper; run `python3 run_all.py --analyse`",
                "match": "",
            })
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero when anything is found")
    parser.add_argument("--skip-manifest", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    root = Path(args.root).resolve()

    findings = (check_anonymity(root) + check_secrets(root)
                + check_required_outputs(root))

    report: Dict[str, Any] = {
        "root": str(root),
        "n_findings": len(findings),
        "findings": findings,
    }
    if not args.skip_manifest:
        manifest = build_manifest(root)
        report["n_files"] = len(manifest)
        manifest_path = root / "MANIFEST.sha256.json"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        logger.info("Manifest: %d files -> %s", len(manifest), manifest_path)

    report_path = root / "results" / "outputs" / "release_audit.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    if findings:
        logger.warning("%d finding(s):", len(findings))
        for finding in findings[:40]:
            logger.warning("  [%s] %s:%s — %s (%s)", finding["kind"],
                           finding["file"], finding["line"], finding["what"],
                           finding["match"])
    else:
        logger.info("No anonymity, secret or missing-output findings.")

    logger.info("Report -> %s", report_path)
    return 1 if (findings and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
