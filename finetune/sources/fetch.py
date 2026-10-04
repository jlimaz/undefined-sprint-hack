"""Download the open datasets used to build the fine-tuning set into data/raw/."""

import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PANORADIO = "http://www.panoradio-sdr.de/wp-content/uploads/"
ARTEMIS_DB = "https://github.com/AresValley/Artemis-DB/releases/download/v74/v74.tar"

# (url, path under data/raw, expected size in bytes or None, expected sha256 or None)
FILES = [
    (PANORADIO + "dataset_panoradio_hf_readme.txt", "panoradio/readme.txt", None, None),
    (PANORADIO + "dataset_panoradio_hf_tags.csv", "panoradio/tags.csv", 3428906, None),
    (PANORADIO + "dataset_panoradio_hf.npy", "panoradio/dataset.npy", 5662310528, None),
    (
        ARTEMIS_DB,
        "artemis/v74.tar",
        302172160,
        "4ef66d5e0eddbc5430cc102f84de0489efd40b979bc6b78abdf259999d09d052",
    ),
]
_CHUNK = 1 << 20


def download(url: str, target: Path, size: int | None) -> None:
    """Fetch url into target, resuming a partial file when the size is known."""
    have = target.stat().st_size if target.exists() else 0
    if size is not None and have == size:
        print(f"have {target.relative_to(ROOT)}")
        return
    if size is None or have > size:
        have = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(request) as response:
        resumed = response.status == 206
        with open(target, "ab" if resumed else "wb") as out:
            done = have if resumed else 0
            reported = done
            while chunk := response.read(_CHUNK):
                out.write(chunk)
                done += len(chunk)
                if done - reported >= 256 * _CHUNK:
                    reported = done
                    total = f" of {size >> 20}" if size else ""
                    print(f"  {target.name}: {done >> 20}{total} MiB", flush=True)
    got = target.stat().st_size
    if size is not None and got != size:
        raise SystemExit(f"{target} is {got} bytes, expected {size}. Run again to resume.")
    print(f"got  {target.relative_to(ROOT)} ({got >> 20} MiB)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    for url, name, size, checksum in FILES:
        target = RAW / name
        download(url, target, size)
        if checksum and sha256(target) != checksum:
            raise SystemExit(f"{target} failed its sha256 check. Delete it and run again.")
    archive = RAW / "artemis" / "v74.tar"
    with tarfile.open(archive) as tar:
        tar.extractall(archive.parent, filter="data")
    print(f"extracted {archive.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
