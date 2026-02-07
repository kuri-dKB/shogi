import json
import zlib
from pathlib import Path

from cshogi import CSA
import cshogi

# 入力
CLEAN_JSONL = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\dataset_train_clean.jsonl"
CSA_DIR = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\csa_out"

# 出力
OUT_DIR = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump"
OUT_TRAIN = "positions_train_s20_e2.jsonl"
OUT_VALID = "positions_valid_s20_e2.jsonl"
OUT_STATS = "positions_stats_s20_e2.txt"

# 設定（必要ならここだけ変更）
SKIP_FIRST_PLIES = 20         # 序盤を捨てたいなら 10 や 20 にする
TAKE_EVERY = 2                # 2にすると2手に1回だけ保存（容量削減）
MAX_POSITIONS_PER_GAME = 0    # 0なら制限なし。例えば200にすると1局200局面まで
VALID_PERCENT = 1             # 1%をvalidにする

# 盤面の一意性や再現性のため、sfen と usi は必ず文字列で保存する


def load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp932", errors="replace")


def load_csa_text(game_id: str) -> str:
    p = Path(CSA_DIR) / f"{game_id}.csa"
    if not p.exists():
        raise FileNotFoundError(f"CSA not found: {p}")
    return load_text(p)


def parse_kifu_from_csa_text(csa_text: str):
    parser = CSA.Parser()
    parsed = parser.parse_str(csa_text)
    # parse_str が list を返す版にも対応
    if isinstance(parsed, list):
        if not parsed:
            raise ValueError("parsed CSA list is empty")
        return parsed[0]
    return parsed


def get_moves(kifu):
    if hasattr(kifu, "moves"):
        return kifu.moves
    if isinstance(kifu, dict) and "moves" in kifu:
        return kifu["moves"]
    raise AttributeError("cannot access moves from kifu")


def try_get_init_sfen(kifu):
    # 取れたら使う。取れないなら標準初期局面。
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


def to_usi(mv_int: int):
    # 無効なmvだと None のことがある
    try:
        return cshogi.move_to_usi(mv_int)
    except Exception:
        return None


def crc32_int(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def is_valid_split(game_id: str) -> bool:
    # だいたい VALID_PERCENT% を valid にする（再現性あり）
    # 例: 1% なら 0..99 のうち 0 が valid
    mod = 100
    threshold = max(1, min(99, VALID_PERCENT))
    return (crc32_int(game_id) % mod) < threshold


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_train = out_dir / OUT_TRAIN
    out_valid = out_dir / OUT_VALID
    out_stats = out_dir / OUT_STATS

    # clean jsonl から game_id を拾う
    game_ids = []
    with Path(CLEAN_JSONL).open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            game_ids.append(obj["game_id"])

    total_games = len(game_ids)
    print(f"games={total_games}")

    train_count = 0
    valid_count = 0
    train_pos = 0
    valid_pos = 0
    fail_games = 0

    with out_train.open("w", encoding="utf-8", newline="\n") as w_train, out_valid.open("w", encoding="utf-8", newline="\n") as w_valid:
        for gi, game_id in enumerate(game_ids, start=1):
            try:
                csa_text = load_csa_text(game_id)
                kifu = parse_kifu_from_csa_text(csa_text)
                moves = get_moves(kifu)

                board = cshogi.Board()
                init_sfen = try_get_init_sfen(kifu)
                if init_sfen:
                    board.set_sfen(init_sfen)
                else:
                    board.reset()

                use_valid = is_valid_split(game_id)
                if use_valid:
                    valid_count += 1
                    w = w_valid
                else:
                    train_count += 1
                    w = w_train

                saved_in_game = 0

                for ply, mv in enumerate(moves, start=1):
                    usi = to_usi(mv)
                    if usi is None:
                        raise ValueError(f"invalid move int at ply={ply}")

                    if not board.is_legal(mv):
                        raise ValueError(f"illegal move at ply={ply} usi={usi}")

                    # 保存条件
                    if ply > SKIP_FIRST_PLIES and ((ply - SKIP_FIRST_PLIES - 1) % TAKE_EVERY == 0):
                        rec = {
                            "game_id": game_id,
                            "ply": ply,
                            "turn": "b" if board.turn == cshogi.BLACK else "w",
                            "sfen": board.sfen(),
                            "move": usi,
                        }
                        w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        saved_in_game += 1
                        if MAX_POSITIONS_PER_GAME and saved_in_game >= MAX_POSITIONS_PER_GAME:
                            # ここで打ち切る（ただし盤面のpushは続けない）
                            break

                    board.push(mv)

                if use_valid:
                    valid_pos += saved_in_game
                else:
                    train_pos += saved_in_game

            except Exception as e:
                fail_games += 1
                # エラーは最初の数件だけ表示
                if fail_games <= 10:
                    print(f"[FAIL] game_id={game_id} {type(e).__name__}: {e}")

            if gi % 200 == 0:
                print(f"progress {gi}/{total_games} train_pos={train_pos} valid_pos={valid_pos} fail_games={fail_games}")

    stats = []
    stats.append(f"CLEAN_JSONL={CLEAN_JSONL}")
    stats.append(f"CSA_DIR={CSA_DIR}")
    stats.append(f"TOTAL_GAMES={total_games}")
    stats.append(f"TRAIN_GAMES={train_count}")
    stats.append(f"VALID_GAMES={valid_count}")
    stats.append(f"FAIL_GAMES={fail_games}")
    stats.append("")
    stats.append(f"SKIP_FIRST_PLIES={SKIP_FIRST_PLIES}")
    stats.append(f"TAKE_EVERY={TAKE_EVERY}")
    stats.append(f"MAX_POSITIONS_PER_GAME={MAX_POSITIONS_PER_GAME}")
    stats.append(f"VALID_PERCENT={VALID_PERCENT}")
    stats.append("")
    stats.append(f"TRAIN_POSITIONS={train_pos}")
    stats.append(f"VALID_POSITIONS={valid_pos}")
    stats.append("")
    stats.append(f"OUT_TRAIN={out_train}")
    stats.append(f"OUT_VALID={out_valid}")

    out_stats.write_text("\n".join(stats), encoding="utf-8")

    print("done.")
    print(f"TRAIN_POSITIONS={train_pos}")
    print(f"VALID_POSITIONS={valid_pos}")
    print(f"FAIL_GAMES={fail_games}")
    print(f"stats={out_stats}")


if __name__ == "__main__":
    main()
