"""Download release attachments with existing gh login, or public author weights."""

import argparse
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path
from ur10e_dp_real.model import sha256

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = "https://diffusion-policy.cs.columbia.edu/data/experiments/image/lift_ph/diffusion_policy_transformer/train_0/checkpoints/epoch=0250-test_mean_score=1.000.ckpt"


def assemble(entry, directory):
    """Verify every chunk and the full original checksum before atomic replacement."""
    dest = Path(directory) / entry["file"]
    if (
        dest.exists()
        and dest.stat().st_size == entry["bytes"]
        and sha256(dest) == entry["sha256"]
    ):
        return dest
    temporary = dest.with_suffix(".assembling")
    try:
        with temporary.open("wb") as output:
            for part in entry["parts"]:
                path = Path(directory) / part["file"]
                if (
                    path.stat().st_size != part["bytes"]
                    or sha256(path) != part["sha256"]
                ):
                    raise ValueError("Chunk checksum mismatch: " + part["file"])
                with path.open("rb") as source:
                    shutil.copyfileobj(source, output)
        if (
            temporary.stat().st_size != entry["bytes"]
            or sha256(temporary) != entry["sha256"]
        ):
            raise ValueError(
                "Reassembled checkpoint checksum mismatch: " + entry["file"]
            )
        temporary.replace(dest)
    finally:
        if temporary.exists():
            temporary.unlink()
    return dest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--author-only", action="store_true")
    p.add_argument("--assemble-only", action="store_true")
    args = p.parse_args()
    manifest = json.loads((ROOT / "weights/manifest.json").read_text())
    if args.author_only:
        dest = ROOT / "weights/author_lift_0250.ckpt"
        temporary = dest.with_suffix(".download")
        with urllib.request.urlopen(AUTHOR, timeout=120) as response, temporary.open(
            "wb"
        ) as output:
            shutil.copyfileobj(response, output)
        temporary.replace(dest)
    elif args.assemble_only:
        for item in manifest:
            assemble(item, ROOT / "weights")
    elif not all(
        (ROOT / "weights" / item["file"]).exists()
        and sha256(ROOT / "weights" / item["file"]) == item["sha256"]
        for item in manifest
    ):
        subprocess.run(
            [
                "gh",
                "release",
                "download",
                "weights-v1",
                "--repo",
                "Ban-TheMind/ur10e_dp_real",
                "--dir",
                str(ROOT / "weights"),
                "--skip-existing",
            ],
            check=True,
        )
        for item in manifest:
            assemble(item, ROOT / "weights")
    for item in manifest:
        dest = ROOT / "weights" / item["file"]
        if args.author_only and item["default"]:
            continue
        if (
            not dest.exists()
            or dest.stat().st_size != item["bytes"]
            or sha256(dest) != item["sha256"]
        ):
            raise ValueError(
                "Weight missing or checksum mismatch: {}".format(item["file"])
            )
        print("Verified", item["file"])


if __name__ == "__main__":
    main()
