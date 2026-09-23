#!/usr/bin/env python3
"""
fetch_artefacts.py — pull the released Phase-1/2 inputs Phase 3 builds on.

Phase 3 reuses the question pool and the persona pools rather than
regenerating them, so the new conditions sit on exactly the items and messages
the reviewers already saw. Those artefacts live in the public release
(the public release repository) and, usually, in a sibling checkout.

The release URL is read from the PLATOS_SHIP_RELEASE_URL environment variable
so that no personal account name is baked into the anonymous artefact. Set it
in .env, or pass --release-url. Without it, only the local search path is used.

It also records a SHA-256 for every input in `inputs_manifest.json`, so the
paper can state which artefact version produced which number, and a reviewer
can check it.

    python3 tools/fetch_artefacts.py                 # local first, then GitHub
    python3 tools/fetch_artefacts.py --source github
    python3 tools/fetch_artefacts.py --verify        # hash only, fetch nothing

Note on what was and was not missing: a local working copy lacked the C4split,
C2het and GPU-probe outputs, but all three are present in the release. Nothing
in the reviewed paper needs to be withdrawn for want of data; the numbers that
do need withdrawing are the perturbed-GSM8K ones, for the separate reason that
their gold labels are invalid (tools/quarantine_perturbed.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("platos_ship3.fetch_artefacts")

# Never hardcode an account name here: the artefact ships for double-blind
# review, and a repository handle identifies the authors.
GITHUB_RAW = os.environ.get("PLATOS_SHIP_RELEASE_URL", "").rstrip("/")

# (destination key in paths.yaml, path inside the release, required?)
ARTEFACTS: List[tuple] = [
    ("question_pool_file",
     "Code_Phase_2/results/processed/question_pool.parquet", True),
    ("anchored_personas_file",
     "Code_Phase_2/results/processed/dumb_personas.parquet", True),
    ("correct_anchored_personas_file",
     "Code_Phase_2/results/processed/correct_anchored_personas.parquet", False),
    ("confidence_personas_file",
     "Code_Phase_2/results/processed/confidence_personas.parquet", False),
    # Protocol-A logs. Phase 3 reads these for the A-vs-B replication, and a
    # working copy can be stale: a local checkout was missing the C4split and
    # C2het trials (2,400 focal trials) that the release does contain. They are
    # fetched into Phase 3's own results/legacy/ rather than overwriting
    # anything, and hashed like every other input.
    ("legacy_phase1_trial_log",
     "results/data/outputs/trial_log.parquet", False),
    ("legacy_phase2_trial_log",
     "Code_Phase_2/results/outputs/trial_log.parquet", False),
    ("legacy_gpu_probe_trials",
     "Code_Phase_2/results/gpu_probe/logprob_probe_trials.parquet", False),
]

# Local fallbacks, relative to the repository root that contains Code_Phase_3.
LOCAL_ROOTS = [
    PROJECT_ROOT.parent,                 # Code/
    PROJECT_ROOT.parent.parent,          # repository root
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_local(relative: str) -> Optional[Path]:
    for root in LOCAL_ROOTS:
        candidate = root / relative
        if candidate.exists():
            return candidate
    return None


def _remote_size(relative: str) -> Optional[int]:
    """Content-Length of a release file, or None when it cannot be read."""
    if not GITHUB_RAW:
        return None
    url = f"{GITHUB_RAW}/{relative}"
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=30) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def _download(relative: str, destination: Path) -> bool:
    if not GITHUB_RAW:
        logger.warning(
            "No release URL configured (set PLATOS_SHIP_RELEASE_URL); cannot "
            "fetch %s remotely.", relative)
        return False
    url = f"{GITHUB_RAW}/{relative}"
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s", url)
        with urllib.request.urlopen(url, timeout=120) as response, \
                open(destination, "wb") as out:
            shutil.copyfileobj(response, out)
        return True
    except Exception as exc:
        logger.warning("Could not download %s: %s", url, exc)
        return False


def fetch(source: str = "auto", verify_only: bool = False) -> Dict[str, Any]:
    import yaml

    with open(PROJECT_ROOT / "config" / "paths.yaml") as handle:
        paths = yaml.safe_load(handle)

    manifest: Dict[str, Any] = {"artefacts": {}, "source": source}
    missing_required: List[str] = []

    for key, relative, required in ARTEFACTS:
        destination = Path(paths[key])
        if not destination.is_absolute():
            destination = PROJECT_ROOT / paths[key]

        if not destination.exists() and not verify_only:
            obtained = False
            if source in ("auto", "local"):
                local = _find_local(relative)
                if local is not None:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(local, destination)
                    logger.info("Copied %s -> %s", local, destination)
                    obtained = True
                    if key.startswith("legacy_"):
                        remote_size = _remote_size(relative)
                        if remote_size and remote_size > destination.stat().st_size:
                            logger.warning(
                                "%s: local copy is %d bytes but the release has "
                                "%d — the working copy is stale. Fetching the "
                                "release version instead.",
                                key, destination.stat().st_size, remote_size)
                            obtained = _download(relative, destination)
            if not obtained and source in ("auto", "github"):
                obtained = _download(relative, destination)
            if not obtained and required:
                missing_required.append(key)
                continue

        if destination.exists():
            manifest["artefacts"][key] = {
                "path": str(destination.relative_to(PROJECT_ROOT))
                if destination.is_relative_to(PROJECT_ROOT) else str(destination),
                "release_path": relative,
                "sha256": sha256_of(destination),
                "bytes": destination.stat().st_size,
            }
        elif required:
            missing_required.append(key)

    manifest["missing_required"] = missing_required
    manifest_path = Path(paths["inputs_manifest_file"])
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / paths["inputs_manifest_file"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    logger.info("Manifest -> %s", manifest_path)
    for key, entry in manifest["artefacts"].items():
        logger.info("  %-34s %s  (%d bytes)", key, entry["sha256"][:16],
                    entry["bytes"])
    if missing_required:
        logger.error("MISSING required artefacts: %s", ", ".join(missing_required))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["auto", "local", "github"],
                        default="auto")
    parser.add_argument("--verify", action="store_true",
                        help="hash what is present; download nothing")
    parser.add_argument("--release-url", default=None,
                        help="raw base URL of the release repository; "
                             "overrides PLATOS_SHIP_RELEASE_URL")
    args = parser.parse_args()

    if args.release_url:
        global GITHUB_RAW
        GITHUB_RAW = args.release_url.rstrip("/")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    manifest = fetch(args.source, args.verify)
    return 1 if manifest["missing_required"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
