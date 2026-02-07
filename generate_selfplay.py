import json
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cshogi
from usi_utils import rotate_usi

# ===== パス =====
POLICY_MODEL_PT = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\policy_model_rel\policy_model.pt"
VOCAB_JSON      = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\policy_model_rel\move_vocab_rel.json"

# Value を使うならここを設定（Value学習で保存したptに合わせる）
USE_VALUE = True
VALUE_MODEL_PT  = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\value_model\value_model.pt"

OUT_SELFPLAY_DIR = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\selfplay_games"
Path(OUT_SELFPLAY_DIR).mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 自己対局設定 =====
NUM_GAMES = 1000
MAX_PLIES = 320
REPEAT_SFEN_LIMIT = 2

RANDOM_OPENING_PLIES = 16

# サンプリング温度（ループ回避で少し上げる）
BASE_TEMPERATURE = 0.9
LOOP_TEMPERATURE = 1.6

# Policyで候補を絞る数 / 最終サンプルのTopK
TOPK_FINAL = 24
TOPK_VALUE_EVAL = 32  # Value評価する候補手数（重いなら 32 でもOK）

# Value の混ぜ具合
LAMBDA_VALUE = 0.6   # 1.0: policyとvalueを同程度に重視
VALUE_CLIP = 1.0      # valueのtanh後をさらにクリップ（保険）
POLICY_MIX_TAU = 8.0   # 2.0〜5.0 あたりから試す。大きいほど value が効く。
SEED = 1234

def sfen_key_no_move_number(sfen: str) -> str:
    # "<board> <turn> <hands> <move_no>" の move_no を捨てる
    # これで同一局面への戻り（千日手系）を検出できる
    p = sfen.split()
    if len(p) < 3:
        return sfen
    return " ".join(p[:3])

# BCEWithLogitsLoss で学習している前提ならこれを使う（推奨）
def value_from_logit(v_logit: torch.Tensor) -> torch.Tensor:
    v = torch.sigmoid(v_logit) * 2.0 - 1.0  # [-1, 1]
    if VALUE_CLIP is not None:
        v = torch.clamp(v, -float(VALUE_CLIP), float(VALUE_CLIP))
    return v

def build_next_sfens_fast(board: cshogi.Board, moves):
    """
    board は現在局面。moves は push できる mv オブジェクトのリスト。
    新しい Board を作らず push/pop で次局面の sfen を作る。
    """
    out = []
    for mv in moves:
        board.push(mv)
        out.append(board.sfen())
        board.pop()  # ここが重要
    return out

def eval_value_for_moves(board, sfen, top_moves, value_model, device, sfen_to_tensor_self_view):
    # 次局面を高速に作る
    next_sfens = build_next_sfens_fast(board, top_moves)

    # バッチテンソル化（numpy->torch は一回）
    xb = np.stack([sfen_to_tensor_self_view(ns) for ns in next_sfens], axis=0)
    xb = torch.from_numpy(xb).to(device, non_blocking=True)

    with torch.inference_mode():
        # fp16 推論（ValueがConv主体なら効くことが多い）
        with torch.autocast(device_type="cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
            v_logit = value_model(xb)  # (B,)

        # BCEWithLogitsLoss 前提なら sigmoid->[-1,1]
        v = torch.sigmoid(v_logit) * 2.0 - 1.0
        # 次局面は相手番self-viewなので符号反転
        v_for_current = (-v).detach().float().cpu().numpy()

    return v_for_current

# ====== ネット定義 ======
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


class SimpleValueNet(nn.Module):
    """
    Value: 盤面 -> スカラー（logit想定）
    学習側が BCEWithLogitsLoss なら、そのまま logit が出てくる想定。
    ここでは tanh で [-1,1] に寄せて使う。
    """
    def __init__(self):
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
        self.fc = nn.Linear(32 * 9 * 9, 1)

    def forward(self, x):
        h = self.body(x)
        h = self.head(h)
        h = torch.flatten(h, 1)
        return self.fc(h).squeeze(1)  # (B,)


# ====== 盤面テンソル化（self-view） ======
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


# ===== ユーティリティ =====
def side_to_move(sfen: str) -> str:
    try:
        return sfen.split()[1]
    except Exception:
        raise ValueError("invalid sfen: " + sfen)

def safe_torch_load(path: str, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def pick_move_from_topk(scored_moves, temperature: float, topk: int):
    scored_moves.sort(key=lambda t: t[0], reverse=True)
    top = scored_moves[:max(1, min(topk, len(scored_moves)))]

    scores = np.array([t[0] for t in top], dtype=np.float64)
    scores = scores / max(1e-6, float(temperature))
    probs = softmax(scores)

    idx = int(np.random.choice(len(top), p=probs))
    return top[idx][1], top[idx][2], float(top[idx][0])


# ====== Selfplay ======
def play_one_game(policy_model, value_model, vocab, game_index: int = 0):
    board = cshogi.Board()
    moves_played = []
    seen = defaultdict(int)
    stop_reason = "unknown"

    for ply in range(1, MAX_PLIES + 1):
        if board.is_game_over():
            stop_reason = "game_over"
            break

        sfen = board.sfen()
        key = sfen_key_no_move_number(sfen)
        seen[key] += 1
        if seen[key] >= REPEAT_SFEN_LIMIT:
            stop_reason = f"repeat_pos_{REPEAT_SFEN_LIMIT}"
            break

        if ply <= RANDOM_OPENING_PLIES:
            legal = list(board.legal_moves)
            if not legal:
                stop_reason = "no_legal_move"
                break
            mv_obj = legal[np.random.randint(len(legal))]
            mv_usi = cshogi.move_to_usi(mv_obj)
            board.push(mv_obj)
            moves_played.append({"usi": mv_usi, "score": None})
            continue

        turn = side_to_move(sfen)
        is_white = (turn == "w")

        x = torch.from_numpy(sfen_to_tensor_self_view(sfen)).unsqueeze(0).to(DEVICE, non_blocking=True)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda"), dtype=torch.float16):
                pol_logits = policy_model(x)[0]

        legal_list = list(board.legal_moves)
        if not legal_list:
            stop_reason = "no_legal_move"
            break

        cand_moves = []
        cand_pol_logits = []
        cand_usi_abs = []
        unknown_abs = 0

        for mv in legal_list:
            usi_abs = cshogi.move_to_usi(mv)
            usi_rel = rotate_usi(usi_abs) if is_white else usi_abs
            idx = vocab.get(usi_rel, None)
            if idx is None:
                unknown_abs += 1
                continue
            cand_moves.append(mv)
            cand_pol_logits.append(pol_logits[idx])
            cand_usi_abs.append(usi_abs)

        if not cand_moves:
            mv_obj = legal_list[np.random.randint(len(legal_list))]
            mv_usi = cshogi.move_to_usi(mv_obj)
            board.push(mv_obj)
            moves_played.append({"usi": mv_usi, "score": None, "fallback": "all_unknown"})
            continue

        cand_pol_logits = torch.stack(cand_pol_logits, dim=0)

        k = min(TOPK_VALUE_EVAL, cand_pol_logits.numel())
        topv, topi = torch.topk(cand_pol_logits, k=k)

        top_moves = [cand_moves[i] for i in topi.tolist()]
        top_usi_abs = [cand_usi_abs[i] for i in topi.tolist()]
        top_pol_logits = topv

        pol_logp = F.log_softmax(top_pol_logits / max(1e-6, float(POLICY_MIX_TAU)), dim=0)

        if (not USE_VALUE) or (value_model is None):
            mix_scores = pol_logp.detach().cpu().numpy()
        else:
            v_for_current = eval_value_for_moves(
                board=board,
                sfen=sfen,
                top_moves=top_moves,
                value_model=value_model,
                device=DEVICE,
                sfen_to_tensor_self_view=sfen_to_tensor_self_view,
            )

            mix_scores = (pol_logp.detach().cpu().numpy()) + float(LAMBDA_VALUE) * v_for_current

        order = np.argsort(-mix_scores)
        order = order[:max(1, min(TOPK_FINAL, len(order)))]

        scores = mix_scores[order]
        temp = LOOP_TEMPERATURE if seen[key] >= 2 else BASE_TEMPERATURE
        scores = scores / max(1e-6, float(temp))
        probs = np.exp(scores - np.max(scores))
        probs = probs / np.sum(probs)

        j = int(np.random.choice(len(order), p=probs))
        pick_idx = int(order[j])

        mv_obj = top_moves[pick_idx]
        mv_usi = top_usi_abs[pick_idx]
        mv_score = float(mix_scores[pick_idx])

        board.push(mv_obj)
        moves_played.append({"usi": mv_usi, "score": mv_score, "unknown_abs": unknown_abs})

    if stop_reason == "unknown":
        stop_reason = "max_plies"

    return {
        "final_sfen": board.sfen(),
        "moves": moves_played,
        "plies": len(moves_played),
        "stop_reason": stop_reason,
    }

def main():
    torch.backends.cudnn.benchmark = True
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    vocab = json.loads(Path(VOCAB_JSON).read_text(encoding="utf-8"))["vocab"]

    # Policy load
    pol_ckpt = safe_torch_load(POLICY_MODEL_PT, DEVICE)
    policy_model = SimplePolicyNet(len(vocab))
    policy_model.load_state_dict(pol_ckpt["model_state"])
    policy_model.to(DEVICE)
    policy_model.eval()

    # Value load（任意）
    value_model = None
    if USE_VALUE:
        v_path = Path(VALUE_MODEL_PT)
        if not v_path.exists():
            raise SystemExit("VALUE_MODEL_PT not found: " + str(v_path))
        v_ckpt = safe_torch_load(str(v_path), DEVICE)

        # 形式を吸収（model_stateがある場合/直state_dictの場合）
        value_model = SimpleValueNet()
        if isinstance(v_ckpt, dict) and "model_state" in v_ckpt:
            value_model.load_state_dict(v_ckpt["model_state"])
        elif isinstance(v_ckpt, dict):
            # まれに state_dict が直で入っているケース
            value_model.load_state_dict(v_ckpt)
        else:
            raise SystemExit("unexpected value ckpt format")

        value_model.to(DEVICE)
        value_model.eval()

    counts = defaultdict(int)
    plies_sum = 0

    t0 = time.time()
    for i in range(1, NUM_GAMES + 1):
        game = play_one_game(policy_model, value_model, vocab, game_index=i)

        ts = int(time.time() * 1000)
        out_path = Path(OUT_SELFPLAY_DIR) / f"selfplay_{ts}_{i:04d}.json"
        out_path.write_text(json.dumps(game, ensure_ascii=False), encoding="utf-8")

        counts[game["stop_reason"]] += 1
        plies_sum += game["plies"]

        if i % 10 == 0 or i == 1:
            avg = plies_sum / i
            elapsed = time.time() - t0
            print(f"[{i}/{NUM_GAMES}] avg_plies={avg:.1f} reasons={dict(counts)} elapsed={elapsed:.1f}s")

    print("done.")
    print("reasons:", dict(counts))
    print("avg_plies:", plies_sum / NUM_GAMES)


if __name__ == "__main__":
    main()
