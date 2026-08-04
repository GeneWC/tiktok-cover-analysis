"""Download TikTok videos from data/new_video_data.xlsx into downloads/.

Uses yt-dlp. Skips video IDs that already exist in downloads/.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "data" / "new_video_data.xlsx"
OUT_DIR = ROOT / "downloads"
LOG = ROOT / "data" / "new_video_download.log"
ERR_LOG = ROOT / "data" / "new_video_download.err.log"

# Prefer system yt-dlp if present; fall back to python -m yt_dlp.
YT_DLP = ["yt-dlp"]


def video_id_from_url(url: str) -> str | None:
    marker = "/video/"
    if marker not in url:
        return None
    return url.split(marker, 1)[1].split("?", 1)[0].strip("/") or None


def already_downloaded(vid: str) -> bool:
    return any(
        (OUT_DIR / f"{vid}{ext}").exists()
        for ext in (".mp4", ".mov", ".m4v", ".webm", ".mkv")
    )


def download_one(url: str, vid: str) -> tuple[bool, str]:
    """Download one video. Filename is downloads/{vid}.%(ext)s."""
    outtmpl = str(OUT_DIR / f"{vid}.%(ext)s")
    cmd = [
        *YT_DLP,
        "--no-playlist",
        "--no-warnings",
        "-f",
        "mp4/best",
        "-o",
        outtmpl,
        "--retries",
        "3",
        "--fragment-retries",
        "3",
        url,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout after 300s"
    except FileNotFoundError:
        return False, "yt-dlp not found on PATH"

    if proc.returncode == 0 and already_downloaded(vid):
        return True, "ok"
    err = (proc.stderr or proc.stdout or "").strip()
    return False, err[-500:] if err else f"exit={proc.returncode}"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(XLSX)
    urls = (
        df["Video Page URL"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    urls = urls[urls.str.startswith("http")].drop_duplicates().tolist()

    jobs: list[tuple[str, str]] = []
    skipped = 0
    for url in urls:
        vid = video_id_from_url(url)
        if not vid:
            continue
        if already_downloaded(vid):
            skipped += 1
            continue
        jobs.append((url, vid))

    print(f"total urls={len(urls)} skip_existing={skipped} to_download={len(jobs)}")
    print(f"logging to {LOG}")

    ok = fail = 0
    with LOG.open("a", encoding="utf-8") as log, ERR_LOG.open(
        "a", encoding="utf-8"
    ) as errlog:
        log.write(f"\n=== start {time.strftime('%Y-%m-%d %H:%M:%S')} jobs={len(jobs)} ===\n")
        log.flush()
        for i, (url, vid) in enumerate(jobs, start=1):
            success, msg = download_one(url, vid)
            line = f"[{i}/{len(jobs)}] {vid} {'OK' if success else 'FAIL'} {msg[:120]}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            if success:
                ok += 1
            else:
                fail += 1
                errlog.write(f"{vid}\t{url}\t{msg}\n")
                errlog.flush()
            # Small pause to reduce rate-limit risk on TikTok.
            time.sleep(1.0)

        summary = f"done ok={ok} fail={fail} skipped={skipped}"
        print(summary)
        log.write(summary + "\n")

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
