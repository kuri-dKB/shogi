import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CSA_DIR = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\csa_out"
OUT_DIR = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump"

OUT_JSONL = "dataset_train.jsonl"
OUT_STATS = "dataset_stats.txt"

# 指し手行: +7776FU / -3334GI など
MOVE_LINE_RE = re.compile(r"^[\+\-]\d{4}[A-Z]{2}($|\s)")
# 終局行: %TORYO / %SENNICHITE など
RESULT_RE = re.compile(r"^%([A-Z_]+)\s*$", re.MULTILINE)

# メタ
EVENT_RE = re.compile(r"^\$EVENT:(.*)$", re.MULTILINE)
OPENING_RE = re.compile(r"^\$OPENING:(.*)$", re.MULTILINE)
START_RE = re.compile(r"^\$START_TIME:(.*)$", re.MULTILINE)
END_RE = re.compile(r"^\$END_TIME:(.*)$", re.MULTILINE)
SENTE_RE = re.compile(r"^N\+(.*)$", re.MULTILINE)
GOTE_RE = re.compile(r"^N\-(.*)$", re.MULTILINE)

# 初期局面（P1〜P9と手番行）
P_LINE_RE = re.compile(r"^P[1-9].*$", re.MULTILINE)
TURN_LINE_RE = re.compile(r"^[\+\-]\s*$", re.MULTILINE)

SKIP_RESULT_TAGS = {"SENNICHITE"}  # 今回は千日手を除外


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp932", errors="replace")


def pick_first(text: str, rx: re.Pattern) -> str:
    m = rx.search(text)
    return m.group(1).strip() if m else ""


def extract_moves(text: str) -> List[str]:
    moves: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if MOVE_LINE_RE.match(line):
            # "+7776FU,T0" みたいにカンマ以降がある場合は前半だけ取りたいことが多い
            # ただし将来使うかもしれないので、まずは行全体を保存せず、基本部分だけ保存
            core = line.split(",")[0].strip()
            moves.append(core)
    return moves


def extract_initial_board(text: str) -> List[str]:
    # 盤面定義P1〜P9と、手番行（+ or -）を順に取り出す
    # CSAは順序に意味があるので、出現順で保存
    lines: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if P_LINE_RE.match(s):
            lines.append(s)
        elif TURN_LINE_RE.match(s):
            lines.append(s)
            # 手番行が出たら初期局面セクションは終わりのことが多い
            # ただし念のため break はせず、後で同じ行が出るケースを避けるなら break にしても良い
            # ここでは break して「初期局面部分だけ」を確実にする
            break
    return lines


def extract_result_tag(text: str) -> str:
    m = RESULT_RE.search(text)
    return m.group(1).strip() if m else ""


def main():
    csa_dir = Path(CSA_DIR)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl_path = out_dir / OUT_JSONL
    out_stats_path = out_dir / OUT_STATS

    files = sorted(csa_dir.glob("*.csa"))
    total = len(files)

    kept = 0
    skipped = 0
    skipped_by_reason: Dict[str, int] = {}

    move_counts: List[int] = []

    with out_jsonl_path.open("w", encoding="utf-8", newline="\n") as w:
        for i, p in enumerate(files, start=1):
            text = safe_read_text(p)

            result_tag = extract_result_tag(text)
            if result_tag in SKIP_RESULT_TAGS:
                skipped += 1
                skipped_by_reason[result_tag] = skipped_by_reason.get(result_tag, 0) + 1
                continue

            moves = extract_moves(text)
            if not moves:
                skipped += 1
                skipped_by_reason["no_moves"] = skipped_by_reason.get("no_moves", 0) + 1
                continue

            initial_board = extract_initial_board(text)

            obj = {
                "game_id": p.stem,
                "event": pick_first(text, EVENT_RE),
                "opening": pick_first(text, OPENING_RE),
                "start_time": pick_first(text, START_RE),
                "end_time": pick_first(text, END_RE),
                "sente": pick_first(text, SENTE_RE),
                "gote": pick_first(text, GOTE_RE),
                "result": result_tag,
                "initial_board": initial_board,
                "moves": moves,
            }

            w.write(json.dumps(obj, ensure_ascii=False) + "\n")

            kept += 1
            move_counts.append(len(moves))

            if i % 1000 == 0:
                print(f"processed {i}/{total} (kept={kept}, skipped={skipped})")

    # stats
    def stats_line(name: str, value) -> str:
        return f"{name}={value}"

    if move_counts:
        min_moves = min(move_counts)
        max_moves = max(move_counts)
        avg_moves = sum(move_counts) / len(move_counts)
    else:
        min_moves = max_moves = avg_moves = 0

    lines: List[str] = []
    lines.append(stats_line("CSA_DIR", str(csa_dir)))
    lines.append(stats_line("TOTAL_FILES", total))
    lines.append(stats_line("KEPT", kept))
    lines.append(stats_line("SKIPPED", skipped))
    lines.append(stats_line("OUTPUT_JSONL", str(out_jsonl_path)))
    lines.append("")
    lines.append(stats_line("MIN_MOVES", min_moves))
    lines.append(stats_line("MAX_MOVES", max_moves))
    lines.append(stats_line("AVG_MOVES", f"{avg_moves:.2f}"))
    lines.append("")
    lines.append("SKIPPED_BREAKDOWN:")
    for k, v in sorted(skipped_by_reason.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"  {k}: {v}")

    out_stats_path.write_text("\n".join(lines), encoding="utf-8")

    print("done.")
    print(out_stats_path)
    print(out_jsonl_path)


if __name__ == "__main__":
    main()
