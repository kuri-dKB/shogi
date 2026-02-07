import json
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import cshogi

# ===== パス =====
MODEL_PT = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\policy_model\policy_model.pt"
VOCAB_JSON = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\policy_model\move_vocab.json"

OUT_SELFPLAY_DIR = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\selfplay_games"
Path(OUT_SELFPLAY_DIR).mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_PLIES = 400
REPEAT_SFEN_LIMIT = 3

# ランダム度（大きいほどランダム）
BASE_TEMPERATURE = 1.2
# ループっぽい時に上げる
LOOP_TEMPERATURE = 2.0
# 上位K手だけから選ぶ
TOPK = 8


# ===== モデル =====
class SimplePolicyNet(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(43, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(32 * 9 * 9, vocab_size)

    def forward(self, x):
        h = self.body(x)
        h = self.head(h)
        h = torch.flatten(h, 1)
        return self.fc(h)


# ===== SFEN -> Tensor =====
PIECE_BASE_TO_KIND = {"P": 0, "L": 1, "N": 2, "S": 3, "G": 4, "B": 5, "R": 6, "K": 7}
PROMOTED_KIND_SHIFT = {"P": 8, "L": 9, "N": 10, "S": 11, "B": 12, "R": 13}

def sfen_to_tensor_self_view(sfen: str) -> np.ndarray:
    parts = sfen.split()
    board_part, turn_part = parts[0], parts[1]
    ranks = board_part.split("/")

    x = np.zeros((43, 9, 9), np.float32)
    turn_is_black = (turn_part == "b")

    for r, rank in enumerate(ranks):
        f = 0
        i = 0
        while i < len(rank):
            c = rank[i]
            if c.isdigit():
                f += int(c)
                i += 1
                continue

            promoted = False
            if c == "+":
                promoted = True
                i += 1
                c = rank[i]

            base = c.upper()
            kind = PROMOTED_KIND_SHIFT[base] if promoted and base in PROMOTED_KIND_SHIFT else PIECE_BASE_TO_KIND[base]
            is_black_piece = c.isupper()
            mine = (is_black_piece == turn_is_black)
            ch = kind if mine else kind + 14

            rr = r if turn_is_black else 8 - r
            ff = f if turn_is_black else 8 - f
            x[ch, rr, ff] = 1.0

            f += 1
            i += 1

    x[28, :, :] = 1.0
    return x


def safe_torch_load(path: str, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def pick_move_from_topk(scored_moves, temperature: float):
    # scored_moves: list[(score, move_obj, usi)]
    scored_moves.sort(key=lambda t: t[0], reverse=True)
    top = scored_moves[:max(1, min(TOPK, len(scored_moves)))]

    scores = np.array([t[0] for t in top], dtype=np.float64)
    # 温度：大きいほど平らになってランダム
    scores = scores / max(1e-6, float(temperature))
    probs = softmax(scores)

    idx = int(np.random.choice(len(top), p=probs))
    return top[idx][1], top[idx][2], float(top[idx][0])


def play_one_game(model, vocab):
    board = cshogi.Board()
    moves_played = []
    sfens_played = []
    seen = defaultdict(int)

    stop_reason = "unknown"

    for ply in range(1, MAX_PLIES + 1):
        if board.is_game_over():
            stop_reason = "game_over"
            break

        sfen = board.sfen()
        sfens_played.append(sfen)

        seen[sfen] += 1
        if seen[sfen] >= REPEAT_SFEN_LIMIT:
            stop_reason = f"repeat_sfen_{REPEAT_SFEN_LIMIT}"
            break

        x = torch.from_numpy(sfen_to_tensor_self_view(sfen)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(x)[0].detach().cpu().numpy()

        scored = []
        for mv in board.legal_moves:
            usi = cshogi.move_to_usi(mv)
            idx = vocab.get(usi, 0)
            score = float(logits[idx])
            scored.append((score, mv, usi))

        if not scored:
            stop_reason = "no_legal_move"
            break

        # ループっぽい局面ほど温度を上げて抜ける
        temp = LOOP_TEMPERATURE if seen[sfen] >= 2 else BASE_TEMPERATURE
        best_move, best_usi, best_score = pick_move_from_topk(scored, temperature=temp)

        board.push(best_move)
        moves_played.append(best_usi)

    if stop_reason == "unknown":
        stop_reason = "max_plies"

    return {
        "start_sfen": cshogi.Board().sfen(),
        "final_sfen": board.sfen(),
        "moves_usi": moves_played,
        "plies": len(moves_played),
        "stop_reason": stop_reason,
    }


def main():
    np.random.seed(int(time.time()) % 2**32)

    vocab = json.loads(Path(VOCAB_JSON).read_text(encoding="utf-8"))["vocab"]
    ckpt = safe_torch_load(MODEL_PT, DEVICE)

    model = SimplePolicyNet(len(vocab))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    game = play_one_game(model, vocab)

    out_path = Path(OUT_SELFPLAY_DIR) / f"selfplay_{int(time.time())}.json"
    out_path.write_text(json.dumps(game, ensure_ascii=False, indent=2), encoding="utf-8")

    print("stop_reason:", game["stop_reason"])
    print("plies:", game["plies"])
    print("saved:", str(out_path))
    print("moves (first 60):", " ".join(game["moves_usi"][:60]))
    if len(game["moves_usi"]) > 60:
        print("...")

if __name__ == "__main__":
    main()
