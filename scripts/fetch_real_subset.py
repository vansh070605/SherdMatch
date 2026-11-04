"""Pull the metadata workbook and a small photo sample out of the 4.7 GB
Daxinzhuang-18K.zip on Zenodo using HTTP range requests, so the full
archive never has to be downloaded."""
import io
import random
import sys
import time
import struct
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

URL = "https://zenodo.org/records/21553799/files/Daxinzhuang-18K.zip?download=1"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "Daxinzhuang-18K"


class HttpFile(io.RawIOBase):
    """Seekable read-only file backed by HTTP range requests."""

    def __init__(self, url):
        self.url = requests.head(url, allow_redirects=True).url
        self.size = int(requests.head(self.url).headers["content-length"])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        for attempt in range(6):  # Zenodo returns sporadic 5xx under load
            try:
                r = requests.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120)
                r.raise_for_status()
                if r.status_code == 206 and len(r.content) == end - self.pos + 1:
                    break
                raise requests.RequestException(f"bad range reply {r.status_code}")
            except requests.RequestException:
                if attempt == 5:
                    raise
                time.sleep(5 * (attempt + 1))
        n = len(r.content)
        b[:n] = r.content
        self.pos += n
        return n


META = OUT / "Daxinzhuang-18K" / "Daxinzhuang_AI_Archaeological_Sherd_Metadata.xlsx"


def exterior_photo_stems(seed=0, n_distractors=300):
    """Exterior-photo stems for every sherd in the expert refit spreadsheet,
    plus random distractor sherds. Photo numbers differ from sherd numbers
    (sherd JD00001 -> photos JD_00001 exterior, JD_00002 interior), so the
    mapping must go through the metadata sheet."""
    import pandas as pd
    meta = pd.read_excel(META, sheet_name="Shandong").iloc[1:]  # row 0 is a field legend
    ext = dict(zip(meta.Sherd_ID.astype(str), meta.Exterior_Photo.astype(str)))
    refit = pd.read_excel(OUT.parent / "expert_confirmed_physical_refit_pairs.xlsx")
    ids = {s for s in refit.stack().astype(str).str.strip() if s in ext}
    others = sorted(set(ext) - ids)
    random.Random(seed).shuffle(others)
    return {ext[s] for s in ids}, {ext[s] for s in others[:n_distractors]}


def main(n_distractors=300, seed=0):
    zf = zipfile.ZipFile(io.BufferedReader(HttpFile(URL), buffer_size=1 << 20))
    names = zf.namelist()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "file_list.txt").write_text("\n".join(names), encoding="utf-8")
    photos = [n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png"))]
    meta = [n for n in names if n.lower().endswith(".xlsx")]
    if not META.exists():
        (OUT / META.relative_to(OUT)).parent.mkdir(parents=True, exist_ok=True)
        META.write_bytes(zf.read(meta[0]))
    # photo stem without view suffix: JD_00121_2.JPG -> JD_00121
    sid = lambda n: "_".join(Path(n).stem.split("_")[:2])
    refit_stems, distractor_stems = exterior_photo_stems(seed, n_distractors)
    refit = [n for n in photos if sid(n) in refit_stems]
    rest = [n for n in photos if sid(n) in distractor_stems]
    print(f"{len(photos)} photos -> {len(refit)} refit photos, {len(rest)} distractor photos")
    todo = [zf.getinfo(n) for n in refit + rest if not (OUT / n).exists()]
    url = zf.fp.raw.url if hasattr(zf.fp, "raw") else HttpFile(URL).url
    with ThreadPoolExecutor(8) as ex:  # one ranged GET per member, in parallel
        for i, _ in enumerate(ex.map(lambda zi: fetch_member(url, zi), todo)):
            if i % 50 == 0:
                print(i, "/", len(todo), flush=True)
    print("done ->", OUT)


def fetch_member(url, zi):
    """Download one zip member with a single range request (local header + data)."""
    start = zi.header_offset
    end = start + 30 + len(zi.filename.encode()) + 2048 + zi.compress_size
    for attempt in range(6):
        try:
            r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            if r.status_code != 206:
                raise requests.RequestException(f"status {r.status_code}")
            break
        except requests.RequestException:
            if attempt == 5:
                raise
            time.sleep(5 * (attempt + 1))
    b = r.content
    if b[:4] != b"PK\x03\x04":
        raise ValueError(f"bad local header for {zi.filename}")
    n, m = struct.unpack("<HH", b[26:30])
    data = b[30 + n + m: 30 + n + m + zi.compress_size]
    if zi.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    if zlib.crc32(data) != zi.CRC:
        raise ValueError(f"CRC mismatch for {zi.filename}")
    dst = OUT / zi.filename
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
