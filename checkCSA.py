import csv
import hashlib
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ここをあなたのフォルダに合わせて固定
CSA_DIR = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\csa_out"

# 出力ファイル（同じフォルダの親に作ります）
OUT_BASENAME = "csa_quality"

# 品質チェックの最低条件（まずは緩め推奨）
MIN_BYTES = 200
MIN_MOVE_LINES = 20

# CSAの指し手行っぽいもの： +7776FU / -3334GI など
MOVE_RE = re.compile(r"^[\+\-]\d{4}[A-Z]{2}($|\s)", re.MULTILINE)

# 終局行： %TORYO, %SENNICHITE, %CHUDAN, %TIME_UP など
RESULT_RE = re.compile(r"^%([A-Z_]+)", re.MULTILINE)

# メタ情報
EVENT_RE = re.compile(r"^\$EVENT:(.*)$", re.MULTILINE)
OPENING_RE = re.compile(r"^\$OPENING:(.*)$", re.MULTILINE)
START_RE = re.compile(r"^\$START_TIME:(.*)$", re.MULTILINE)
END_RE = re.compile(r"^\$END_TIME:(.*)$", re.MULTILINE)
SENTE_RE = re.compile(r"^N\+(.*)$", re.MULTILINE)
GOTE_RE = re.compile(r"^N\-(.*)$", re.MULTILINE)


@dataclass
class FileCheck:
    path: str
    size_bytes: int
    ok: bool
    reasons: List[str]
    has_v2: bool
    has_board: bool
    has_turn: bool
    move_lines: int
    result_tag: str
    event: str
    opening: str
    start_time: str
    end_time: str
    sente: str
    gote: str
    sha1: str


def safe_read_text(path: Path) -> str:
    # CSAは基本ASCII寄りだが、日本語メタもあるのでUTF-8優先、ダメならcp932も試す
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp932", errors="replace")


def compute_sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def normalize_for_hash(text: str) -> str:
    # 余計な改行差を吸収する程度の正規化（内容同一判定用）
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    return t.strip()


def check_one_file(path: Path) -> FileCheck:
    raw_bytes = path.read_bytes()
    size_bytes = len(raw_bytes)
    sha1 = compute_sha1_bytes(raw_bytes)

    text = safe_read_text(path)
    norm = normalize_for_hash(text)

    reasons: List[str] = []

    has_v2 = ("V2." in norm[:200])  # 先頭付近にあればOK扱い
    if not has_v2:
        reasons.append("no_V2_header")

    # 盤面行P1〜P9があるか（全部でなくても最低限あるか）
    has_board = any(f"\nP{i}" in ("\n" + norm) for i in range(1, 10))
    if not has_board:
        reasons.append("no_board_lines")

    # 手番行（単独の + / - 行）があるか
    # 例： ... P9 ... の後に "\n+\n" のように出る
    has_turn = bool(re.search(r"(?m)^[\+\-]\s*$", norm))
    if not has_turn:
        reasons.append("no_turn_line")

    move_lines = len(MOVE_RE.findall(norm))
    if move_lines < MIN_MOVE_LINES:
        reasons.append(f"too_few_moves(<{MIN_MOVE_LINES})")

    if size_bytes < MIN_BYTES:
        reasons.append(f"too_small(<{MIN_BYTES}bytes)")

    # 終局タグ
    m = RESULT_RE.search(norm)
    result_tag = m.group(1) if m else ""

    # メタ情報（無ければ空）
    def pick(rx) -> str:
        mm = rx.search(norm)
        return (mm.group(1).strip() if mm else "")

    event = pick(EVENT_RE)
    opening = pick(OPENING_RE)
    start_time = pick(START_RE)
    end_time = pick(END_RE)
    sente = pick(SENTE_RE)
    gote = pick(GOTE_RE)

    ok = (len(reasons) == 0)

    return FileCheck(
        path=str(path),
        size_bytes=size_bytes,
        ok=ok,
        reasons=reasons,
        has_v2=has_v2,
        has_board=has_board,
        has_turn=has_turn,
        move_lines=move_lines,
        result_tag=result_tag,
        event=event,
        opening=opening,
        start_time=start_time,
        end_time=end_time,
        sente=sente,
        gote=gote,
        sha1=sha1,
    )


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    csa_dir = Path(CSA_DIR)
    if not csa_dir.exists():
        raise SystemExit(f"Not found: {csa_dir}")

    out_dir = csa_dir.parent
    summary_path = out_dir / f"{OUT_BASENAME}_summary.txt"
    details_path = out_dir / f"{OUT_BASENAME}_details.csv"
    bad_path = out_dir / f"{OUT_BASENAME}_bad.csv"
    dup_path = out_dir / f"{OUT_BASENAME}_duplicates.csv"

    files = sorted(csa_dir.glob("*.csa"))
    total = len(files)
    print(f"Found {total} .csa files")

    results: List[FileCheck] = []
    reason_counter = Counter()
    result_counter = Counter()
    event_counter = Counter()
    opening_counter = Counter()

    # 重複判定（内容SHA）
    content_hash_map: Dict[str, List[str]] = defaultdict(list)

    for i, p in enumerate(files, start=1):
        fc = check_one_file(p)
        results.append(fc)

        if not fc.ok:
            for r in fc.reasons:
                reason_counter[r] += 1

        if fc.result_tag:
            result_counter[fc.result_tag] += 1
        if fc.event:
            event_counter[fc.event] += 1
        if fc.opening:
            opening_counter[fc.opening] += 1

        # 内容での重複検出（sha1は生bytes、ここでは内容正規化shaも使う）
        # raw sha1だけでも重複検出になるが、改行差があり得るので正規化shaも作る
        norm = normalize_for_hash(safe_read_text(p))
        norm_sha1 = hashlib.sha1(norm.encode("utf-8", errors="replace")).hexdigest()
        content_hash_map[norm_sha1].append(str(p))

        if i % 500 == 0:
            print(f"  checked {i}/{total}")

    ok_count = sum(1 for r in results if r.ok)
    bad_count = total - ok_count

    # move_linesの分布（ざっくり）
    move_bins = Counter()
    for r in results:
        ml = r.move_lines
        if ml < 30:
            move_bins["<30"] += 1
        elif ml < 60:
            move_bins["30-59"] += 1
        elif ml < 100:
            move_bins["60-99"] += 1
        elif ml < 150:
            move_bins["100-149"] += 1
        else:
            move_bins["150+"] += 1

    # 重複候補
    dup_groups = [(h, paths) for h, paths in content_hash_map.items() if len(paths) >= 2]
    dup_groups.sort(key=lambda x: len(x[1]), reverse=True)

    # CSV: details
    details_rows: List[Dict[str, str]] = []
    bad_rows: List[Dict[str, str]] = []
    for r in results:
        row = {
            "path": r.path,
            "size_bytes": str(r.size_bytes),
            "ok": "1" if r.ok else "0",
            "reasons": "|".join(r.reasons),
            "has_v2": "1" if r.has_v2 else "0",
            "has_board": "1" if r.has_board else "0",
            "has_turn": "1" if r.has_turn else "0",
            "move_lines": str(r.move_lines),
            "result_tag": r.result_tag,
            "event": r.event,
            "opening": r.opening,
            "start_time": r.start_time,
            "end_time": r.end_time,
            "sente": r.sente,
            "gote": r.gote,
            "sha1": r.sha1,
        }
        details_rows.append(row)
        if not r.ok:
            bad_rows.append(row)

    fieldnames = [
        "path", "size_bytes", "ok", "reasons",
        "has_v2", "has_board", "has_turn",
        "move_lines", "result_tag",
        "event", "opening", "start_time", "end_time",
        "sente", "gote",
        "sha1",
    ]
    write_csv(details_path, details_rows, fieldnames)
    write_csv(bad_path, bad_rows, fieldnames)

    # CSV: duplicates
    dup_rows: List[Dict[str, str]] = []
    for h, paths in dup_groups:
        for p in paths:
            dup_rows.append({"norm_sha1": h, "path": p, "group_size": str(len(paths))})
    write_csv(dup_path, dup_rows, ["norm_sha1", "group_size", "path"])

    # summary.txt
    lines: List[str] = []
    lines.append(f"CSA_DIR={csa_dir}")
    lines.append(f"TOTAL={total}")
    lines.append(f"OK={ok_count}")
    lines.append(f"BAD={bad_count}")
    lines.append("")
    lines.append("BAD_REASONS (count):")
    for k, v in reason_counter.most_common():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("MOVE_LINES_BINS (count):")
    for k in ["<30", "30-59", "60-99", "100-149", "150+"]:
        lines.append(f"  {k}: {move_bins.get(k, 0)}")
    lines.append("")
    lines.append("RESULT_TAG_TOP20 (count):")
    for k, v in result_counter.most_common(20):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("EVENT_TOP20 (count):")
    for k, v in event_counter.most_common(20):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("OPENING_TOP20 (count):")
    for k, v in opening_counter.most_common(20):
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append(f"DUPLICATE_GROUPS={len(dup_groups)}")
    if dup_groups:
        lines.append("DUPLICATE_GROUPS_TOP10 (group_size):")
        for h, paths in dup_groups[:10]:
            lines.append(f"  size={len(paths)} hash={h} example={paths[0]}")
    lines.append("")
    lines.append(f"DETAILS_CSV={details_path}")
    lines.append(f"BAD_CSV={bad_path}")
    lines.append(f"DUP_CSV={dup_path}")

    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("Done.")
    print(f"Summary: {summary_path}")
    print(f"Details: {details_path}")
    print(f"Bad:     {bad_path}")
    print(f"Dup:     {dup_path}")


if __name__ == "__main__":
    main()
