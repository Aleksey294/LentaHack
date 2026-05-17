from __future__ import annotations

import html
import re
import sys
import urllib.request
from pathlib import Path

ROOT_ID = "1XRrRB7y66RU4lxZiH7a6H_b8fOvKgOQl"
ROOT_URL = f"https://drive.google.com/drive/folders/{ROOT_ID}"
OUT = Path("data/raw")
LABELS = Path("data/labels")
VIDEOS = Path("data/videos")
OUT.mkdir(parents=True, exist_ok=True)
LABELS.mkdir(parents=True, exist_ok=True)
VIDEOS.mkdir(parents=True, exist_ok=True)


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def download_file(file_id: str, dst: Path) -> None:
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    dst.write_bytes(data)
    print(f"downloaded {dst} ({len(data)} bytes)")


def entries(page: str):
    # Google Drive exposes an escaped _DRIVE_ivd payload; this regex is enough for public folder listings.
    text = html.unescape(page).encode("utf-8").decode("unicode_escape", errors="ignore")
    pattern = re.compile(r'\["([A-Za-z0-9_-]{20,})",\["[A-Za-z0-9_-]+"\],"([^"]+)","([^"]+)"')
    seen = set()
    for file_id, name, mime in pattern.findall(text):
        key = (file_id, name)
        if key in seen:
            continue
        seen.add(key)
        yield {"id": file_id, "name": name, "mime": mime.replace("\\/", "/")}


def main() -> int:
    root = fetch(ROOT_URL)
    (OUT / "root.html").write_text(root, encoding="utf-8")
    root_entries = list(entries(root))
    print("root entries:")
    for e in root_entries:
        print(e)

    for e in root_entries:
        if e["name"] == "sample.csv":
            download_file(e["id"], Path("sample.csv"))
        if e["mime"] == "application/vnd.google-apps.folder":
            folder_html = fetch(f"https://drive.google.com/drive/folders/{e['id']}")
            folder_path = OUT / f"{e['name']}.html"
            folder_path.write_text(folder_html, encoding="utf-8")
            for child in entries(folder_html):
                print(e["name"], child)
                if child["name"].lower().endswith(".csv"):
                    download_file(child["id"], LABELS / child["name"])
                if "--videos" in sys.argv and child["name"].lower().endswith(".mp4") and e["name"] != "Unlabeled":
                    download_file(child["id"], VIDEOS / child["name"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
