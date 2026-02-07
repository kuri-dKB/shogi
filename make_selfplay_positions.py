import json
import random
from pathlib import Path
import cshogi
from usi_utils import rotate_usi

SELFPLAY_DIR = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\selfplay_games"
OUT_JSONL = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\positions_selfplay_value.jsonl"
VOCAB_JSON = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\policy_model_rel\move_vocab_rel.json"

CAP_PLIES_PER_GAME = 260
MIN_PLIES = 40
TAKE_EVERY = 28

SEED = 1234

def push_usi(board, usi: str):
    if hasattr(board, "push_usi"):
        board.push_usi(usi)
        return
    for mv in board.legal_moves:
        if cshogi.move_to_usi(mv) == usi:
            board.push(mv)
            return
    raise ValueError("cannot push usi: " + usi)

def load_vocab(path: str):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return obj["vocab"]

def side_to_move(sfen: str) -> str:
    parts = sfen.split()
    if len(parts) < 2:
        raise ValueError("bad sfen: " + sfen)
    return parts[1]  # "b" or "w"

def winner_from_final_sfen_turn_only(final_sfen: str) -> str:
    # stop_reason == "game_over" のときだけ呼ぶ想定
    b = cshogi.Board()
    b.set_sfen(final_sfen)
    loser = b.turn
    winner = 1 - loser
    return "b" if winner == cshogi.BLACK else "w"

def main():
    random.seed(SEED)

    vocab = load_vocab(VOCAB_JSON)

    in_dir = Path(SELFPLAY_DIR)
    files = sorted(in_dir.glob("selfplay_*.json"))
    if not files:
        raise SystemExit("no selfplay files: " + str(in_dir))

    out = Path(OUT_JSONL)
    out.parent.mkdir(parents=True, exist_ok=True)

    stop_reason_counts = {}
    z_counts = {1: 0, -1: 0}
    winner_counts = {"b": 0, "w": 0}

    games_total = 0
    games_used = 0
    skipped_non_game_over = 0
    skipped_short = 0
    failed = 0

    positions_written = 0
    dropped_unknown_moves = 0

    with out.open("w", encoding="utf-8") as f_out:
        for fp in files:
            games_total += 1
            try:
                obj = json.loads(fp.read_text(encoding="utf-8"))
                sr = obj.get("stop_reason", "missing")
                stop_reason_counts[sr] = stop_reason_counts.get(sr, 0) + 1

                # 勝敗確定してないゲームは捨てる
                if sr != "game_over":
                    skipped_non_game_over += 1
                    continue

                final_sfen = obj.get("final_sfen", "")
                if not final_sfen:
                    skipped_non_game_over += 1
                    continue

                winner = winner_from_final_sfen_turn_only(final_sfen)
                if winner not in ("b", "w"):
                    skipped_non_game_over += 1
                    continue
                winner_counts[winner] += 1

                moves = obj.get("moves", [])
                usis = [m["usi"] for m in moves if isinstance(m, dict) and "usi" in m]

                if len(usis) < MIN_PLIES:
                    skipped_short += 1
                    continue

                board = cshogi.Board()
                limit = min(len(usis), CAP_PLIES_PER_GAME)

                # ★重要: ゲームごとに間引き開始位置をランダムにする
                # これで手番の偏り（常に先手番だけ）が消える
                offset = random.randrange(TAKE_EVERY) if TAKE_EVERY > 1 else 0

                for i in range(limit):
                    mv = usis[i]
                    take = ((i + offset) % TAKE_EVERY == 0)

                    if take:
                        # 現在の局面のSFENから手番を取得
                        sfen = board.sfen()
                        stm = side_to_move(sfen)  # "b" or "w"
                        
                        # vocabチェック用にRelative化
                        if stm == "w":
                            mv_check = rotate_usi(mv)
                        else:
                            mv_check = mv

                        if mv_check in vocab:
                            z = 1 if stm == winner else -1
                            # 書き出す move は Absolute のままでもよい
                            # （train_policy 側は読み込み時に rotate するようになったため）
                            rec = {"sfen": sfen, "move": mv, "z": z}
                            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            positions_written += 1
                            z_counts[z] += 1
                        else:
                            dropped_unknown_moves += 1

                    push_usi(board, mv)

                games_used += 1

            except Exception as e:
                failed += 1
                print("[FAIL]", fp.name, type(e).__name__, e)

    total_z = z_counts[1] + z_counts[-1]
    maj = max(z_counts[1], z_counts[-1]) if total_z else 0
    maj_ratio = (maj / total_z) if total_z else 0.0

    total_w = winner_counts["b"] + winner_counts["w"]
    w_maj = max(winner_counts["b"], winner_counts["w"]) if total_w else 0
    w_maj_ratio = (w_maj / total_w) if total_w else 0.0

    print("DONE")
    print("games_total =", games_total)
    print("games_used  =", games_used)
    print("skipped_non_game_over =", skipped_non_game_over)
    print("skipped_short =", skipped_short)
    print("failed =", failed)
    print("positions_written =", positions_written)
    print("stop_reason_counts =", stop_reason_counts)
    print("dropped_unknown_moves =", dropped_unknown_moves)
    print("winner_counts =", winner_counts)
    print("winner_majority_ratio =", w_maj_ratio)
    print("z_counts =", z_counts)
    print("z_majority_ratio =", maj_ratio)
    print("out =", str(out))

if __name__ == "__main__":
    main()
