import json
import re
from pathlib import Path

from cshogi import CSA
import cshogi

DATASET_JSONL = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\dataset_train.jsonl"
CSA_DIR = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\csa_out"
OUT_CLEAN_JSONL = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\dataset_train_clean.jsonl"
OUT_FAIL_TXT = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\csa_verify_failures.txt"

# デバッグ用：CSAの指し手行だけ抽出（+7776FU など）
MOVE_LINE_RE = re.compile(r"^[\+\-]\d{4}[A-Z]{2}(\,.*)?$")


def load_csa_text(game_id: str) -> str:
    p = Path(CSA_DIR) / f"{game_id}.csa"
    if not p.exists():
        raise FileNotFoundError(f"CSA file not found: {p}")
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="cp932", errors="replace")


def extract_csa_move_lines(csa_text: str):
    lines = []
    for ln in csa_text.splitlines():
        s = ln.strip()
        if MOVE_LINE_RE.match(s):
            lines.append(s.split(",")[0])
    return lines


def get_first_kifu(parsed):
    if isinstance(parsed, list):
        if not parsed:
            raise ValueError("parsed list is empty")
        return parsed[0]
    return parsed


def get_moves(kifu):
    if hasattr(kifu, "moves"):
        return kifu.moves
    if isinstance(kifu, dict) and "moves" in kifu:
        return kifu["moves"]
    raise AttributeError("cannot access moves from kifu")


def try_get_init_sfen(kifu):
    # できるだけ広く拾う（取れなければNone）
    if hasattr(kifu, "init_board") and kifu.init_board is not None and hasattr(kifu.init_board, "sfen"):
        try:
            v = kifu.init_board.sfen()
            if isinstance(v, str) and v.strip():
                return v.strip()
        except Exception:
            pass

    for a in ["init_sfen", "initial_sfen", "start_sfen", "sfen"]:
        if hasattr(kifu, a):
            v = getattr(kifu, a)
            if callable(v):
                try:
                    v = v()
                except TypeError:
                    pass
            if isinstance(v, str) and v.strip():
                return v.strip()

    if isinstance(kifu, dict):
        for k in ["init_sfen", "initial_sfen", "start_sfen", "sfen"]:
            v = kifu.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

    return None


def mv_to_usi_safe(mv: int):
    # mvが無効だと None になることがある
    try:
        return cshogi.move_to_usi(mv)
    except Exception:
        return None


def verify_one_game_by_csa(game_id: str) -> None:
    csa_text = load_csa_text(game_id)

    parser = CSA.Parser()
    parsed = parser.parse_str(csa_text)
    kifu = get_first_kifu(parsed)

    moves = get_moves(kifu)
    csa_lines = extract_csa_move_lines(csa_text)

    board = cshogi.Board()
    init_sfen = try_get_init_sfen(kifu)
    if init_sfen:
        board.set_sfen(init_sfen)
    else:
        board.reset()

    for idx, mv in enumerate(moves, start=1):
        usi = mv_to_usi_safe(mv)
        if usi is None:
            # mv が無効
            csa_line = csa_lines[idx - 1] if idx - 1 < len(csa_lines) else "(no_csa_line)"
            raise ValueError(f"invalid move int at ply={idx} csa_line={csa_line}")

        if not board.is_legal(mv):
            csa_line = csa_lines[idx - 1] if idx - 1 < len(csa_lines) else "(no_csa_line)"
            raise ValueError(f"illegal move at ply={idx} usi={usi} csa_line={csa_line} sfen_before={board.sfen()}")

        board.push(mv)


def main():
    src = Path(DATASET_JSONL)
    out = Path(OUT_CLEAN_JSONL)
    fail_txt = Path(OUT_FAIL_TXT)

    failures = []  # (game_id, reason)

    # 1) まず全行を読みつつ検証し、落ちたgame_idを記録
    rows = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append(obj)

    total = len(rows)
    ok = 0
    fail = 0

    for i, obj in enumerate(rows, start=1):
        game_id = obj["game_id"]
        try:
            verify_one_game_by_csa(game_id)
            ok += 1
        except Exception as e:
            fail += 1
            failures.append((game_id, f"{type(e).__name__}: {e}"))
            # 進捗表示を兼ねて、最初の数件だけ標準出力に出す
            if fail <= 10:
                print(f"[FAIL] {i}/{total} game_id={game_id} {type(e).__name__}: {e}")

        if i % 500 == 0:
            print(f"progress {i}/{total} ok={ok} fail={fail}")

    # 2) 落ちたものを除外して clean JSONL を作る
    fail_set = set(g for g, _ in failures)

    with out.open("w", encoding="utf-8", newline="\n") as w:
        for obj in rows:
            if obj["game_id"] in fail_set:
                continue
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # 3) failure一覧を保存
    with fail_txt.open("w", encoding="utf-8") as w:
        w.write(f"TOTAL={total}\nOK={ok}\nFAIL={fail}\n\n")
        for gid, reason in failures:
            w.write(f"{gid}\t{reason}\n")

    print("done.")
    print(f"TOTAL={total} OK={ok} FAIL={fail}")
    print(f"CLEAN_JSONL={out}")
    print(f"FAIL_LIST={fail_txt}")


if __name__ == "__main__":
    main()
