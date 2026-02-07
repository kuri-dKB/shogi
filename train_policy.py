import argparse
import json
import math
import time
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import Dataset, DataLoader

from usi_utils import rotate_usi




# ====== パス（あなたの環境） ======
POSITIONS_TRAIN = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\positions_train_mix.jsonl"
POSITIONS_VALID = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\positions_valid_s20_e2.jsonl"
OUT_DIR = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\policy_model_rel"

VOCAB_JSON = "move_vocab_rel.json"
MODEL_PT = "policy_model_rel.pt"
STATS_TXT = "train_stats.txt"


# ====== SFEN -> 入力テンソル（cshogi API 不要） ======
# 43面 x 9 x 9
#  0-13: 自分の駒14種（成り含む）
# 14-27: 相手の駒14種
# 28: 手番（常に1.0; 自分視点に正規化）
# 29-35: 自分の持ち駒7種（P,L,N,S,G,B,R）
# 36-42: 相手の持ち駒7種

PIECE_BASE_TO_KIND = {
    "P": 0,
    "L": 1,
    "N": 2,
    "S": 3,
    "G": 4,
    "B": 5,
    "R": 6,
    "K": 7,
}

# 成りは + の有無で判断して、kind を 8..13 に寄せる（K,Gは成らない）
PROMOTED_KIND_SHIFT = {
    "P": 8,   # +P
    "L": 9,   # +L
    "N": 10,  # +N
    "S": 11,  # +S
    "B": 12,  # +B
    "R": 13,  # +R
}

HAND_ORDER = ["P", "L", "N", "S", "G", "B", "R"]


def parse_sfen(sfen: str):
    # sfen: "<board> <turn> <hands> <move_no>"
    parts = sfen.strip().split()
    if len(parts) < 4:
        raise ValueError(f"bad sfen: {sfen}")
    board_part, turn_part, hands_part, move_no = parts[0], parts[1], parts[2], parts[3]
    return board_part, turn_part, hands_part, move_no


def parse_board_part(board_part: str):
    # 9 ranks separated by '/'
    ranks = board_part.split("/")
    if len(ranks) != 9:
        raise ValueError(f"bad board part: {board_part}")
    return ranks


def piece_token_to_kind14(token: str) -> int:
    # token: "P" "p" "+P" "+p" etc
    promoted = False
    if token.startswith("+"):
        promoted = True
        token = token[1:]

    base = token.upper()
    if base not in PIECE_BASE_TO_KIND:
        raise ValueError(f"unknown piece token: {token}")

    if promoted and base in PROMOTED_KIND_SHIFT:
        return PROMOTED_KIND_SHIFT[base]
    else:
        return PIECE_BASE_TO_KIND[base]


def parse_hands(hands_part: str):
    # hands: "-" or like "2Rb3p" etc
    # returns dict counts for black(uppercase) and white(lowercase)
    b = {k: 0 for k in HAND_ORDER}
    w = {k: 0 for k in HAND_ORDER}
    if hands_part == "-" or hands_part == "":
        return b, w

    i = 0
    n = len(hands_part)
    while i < n:
        # optional number
        j = i
        while j < n and hands_part[j].isdigit():
            j += 1
        cnt = 1
        if j > i:
            cnt = int(hands_part[i:j])
        if j >= n:
            raise ValueError(f"bad hands: {hands_part}")
        pc = hands_part[j]
        i = j + 1

        base = pc.upper()
        if base not in HAND_ORDER:
            raise ValueError(f"bad hands piece: {pc} in {hands_part}")

        if pc.isupper():
            b[base] += cnt
        else:
            w[base] += cnt

    return b, w


def sfen_to_tensor_self_view(sfen: str) -> np.ndarray:
    board_part, turn_part, hands_part, _ = parse_sfen(sfen)
    ranks = parse_board_part(board_part)

    x = np.zeros((43, 9, 9), dtype=np.float32)

    # turn_part: "b" or "w"
    # self_view: 常に「手番側=自分」
    # 手番が w のときは盤面を180度回転して、色の所属も自分/相手で解釈する
    turn_is_black = (turn_part == "b")

    # 盤面
    # SFEN rank は上から（9段目->1段目）。各rank内は左から（9筋->1筋）。
    for r, rank_str in enumerate(ranks):
        file_idx = 0
        i = 0
        while i < len(rank_str):
            ch = rank_str[i]
            if ch.isdigit():
                file_idx += int(ch)
                i += 1
                continue

            promoted = False
            if ch == "+":
                promoted = True
                i += 1
                if i >= len(rank_str):
                    raise ValueError(f"bad promoted token in rank: {rank_str}")
                ch = rank_str[i]

            token = ("+" if promoted else "") + ch

            # 駒の色（SFENでは大文字=先手, 小文字=後手）
            is_black_piece = ch.isupper()

            # 自分駒か相手駒か
            # 手番が b のとき: 黒駒が自分
            # 手番が w のとき: 白駒が自分（is_black_piece=False が自分）
            mine = (is_black_piece == turn_is_black)

            kind = piece_token_to_kind14(token)
            ch_idx = kind if mine else (14 + kind)

            # 座標（rank=r, file=file_idx）を self-view に合わせる
            rr = r
            ff = file_idx
            if not turn_is_black:
                rr = 8 - rr
                ff = 8 - ff

            x[ch_idx, rr, ff] = 1.0

            file_idx += 1
            i += 1

        if file_idx != 9:
            raise ValueError(f"rank width != 9: {rank_str} width={file_idx}")

    # 手番面（常に自分番に正規化しているので 1.0 固定）
    x[28, :, :] = 1.0

    # 持ち駒
    b_hand, w_hand = parse_hands(hands_part)
    my_hand = b_hand if turn_is_black else w_hand
    op_hand = w_hand if turn_is_black else b_hand

    denom = 18.0  # 正規化

    for i, k in enumerate(HAND_ORDER):
        x[29 + i, :, :] = float(my_hand[k]) / denom
        x[36 + i, :, :] = float(op_hand[k]) / denom

    return x


# ====== move 辞書 ======
def build_move_vocab(train_jsonl: str, out_vocab_json: str, max_unique: int = 200000) -> None:
    moves = {}
    count = 0
    with Path(train_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # sfen から手番を取得
            sfen = obj["sfen"]
            _, turn_part, _, _ = parse_sfen(sfen)
            mv_abs = obj["move"]
            
            # 手番が後手なら回転して「自分視点」にする
            if turn_part == "w":
                mv = rotate_usi(mv_abs)
            else:
                mv = mv_abs
            
            moves[mv] = moves.get(mv, 0) + 1
            count += 1
            if count % 200000 == 0:
                print(f"scanned={count} unique_moves={len(moves)}")
            if len(moves) >= max_unique:
                print("hit max_unique; stopping scan")
                break

    items = sorted(moves.items(), key=lambda x: (-x[1], x[0]))

    vocab = {"__UNK__": 0}
    for i, (mv, _) in enumerate(items, start=1):
        vocab[mv] = i

    out = {
        "vocab": vocab,
        "size": len(vocab),
        "scanned_lines": count,
        "unique_moves": len(items),
        "top20": items[:20],
    }

    Path(out_vocab_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_vocab_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved vocab: {out_vocab_json}")
    print(f"vocab_size={len(vocab)}")


def load_vocab(vocab_json: str) -> Dict[str, int]:
    obj = json.loads(Path(vocab_json).read_text(encoding="utf-8"))
    return obj["vocab"]


# ====== Dataset ======
class PositionsDataset(Dataset):
    def __init__(self, jsonl_path: str, vocab: Dict[str, int]):
        self.path = Path(jsonl_path)
        self.vocab = vocab

        self.offsets = []
        with self.path.open("rb") as f:
            pos = 0
            while True:
                line = f.readline()
                if not line:
                    break
                self.offsets.append(pos)
                pos = f.tell()
        print(f"indexed lines={len(self.offsets)} file={self.path}")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int):
        with self.path.open("rb") as f:
            f.seek(self.offsets[idx])
            line = f.readline().decode("utf-8")
        obj = json.loads(line)

        sfen = obj["sfen"]
        mv_abs = obj["move"]

        # 手番取得
        _, turn_part, _, _ = parse_sfen(sfen)
        
        # 後手なら回転（Relative化）
        if turn_part == "w":
            mv_rel = rotate_usi(mv_abs)
        else:
            mv_rel = mv_abs

        y = self.vocab.get(mv_rel, 0)
        x = sfen_to_tensor_self_view(sfen)

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# ====== Model ======
class SimplePolicyNet(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(43, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(32 * 9 * 9, vocab_size)

    def forward(self, x):
        h = self.body(x)
        h = self.head(h)
        h = torch.flatten(h, 1)
        return self.fc(h)


@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    log_every: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    init_model: str = ""   # ★追加



def accuracy_topk(logits: torch.Tensor, y: torch.Tensor, k: int = 1) -> float:
    with torch.no_grad():
        top = torch.topk(logits, k=k, dim=1).indices
        y2 = y.view(-1, 1)
        ok = (top == y2).any(dim=1).float().mean().item()
        return ok


def train_one_epoch(model, loader, optimizer, device, scaler, log_every=200):
    model.train()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_n = 0
    acc1_sum = 0.0
    acc5_sum = 0.0

    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with amp.autocast(enabled=(device.type == "cuda")):
            logits = model(x)
            loss = ce(logits, y)

        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        acc1_sum += accuracy_topk(logits, y, k=1) * bs
        acc5_sum += accuracy_topk(logits, y, k=5) * bs

        if step % log_every == 0:
            print(f"step={step} loss={total_loss/total_n:.4f} acc1={acc1_sum/total_n:.4f} acc5={acc5_sum/total_n:.4f}")
    return total_loss / total_n, acc1_sum / total_n, acc5_sum / total_n


def eval_one_epoch(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_n = 0
    acc1_sum = 0.0
    acc5_sum = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = ce(logits, y)

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_n += bs
            acc1_sum += accuracy_topk(logits, y, k=1) * bs
            acc5_sum += accuracy_topk(logits, y, k=5) * bs

    return total_loss / total_n, acc1_sum / total_n, acc5_sum / total_n


def run_train(cfg: TrainConfig, vocab_json: str):
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab(vocab_json)
    vocab_size = len(vocab)
    print(f"vocab_size={vocab_size}")

    train_ds = PositionsDataset(POSITIONS_TRAIN, vocab)
    valid_ds = PositionsDataset(POSITIONS_VALID, vocab)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.startswith("cuda")),
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.startswith("cuda")),
        drop_last=False,
    )

    device = torch.device(cfg.device)
    model = SimplePolicyNet(vocab_size).to(device)

    # 保存先（ベストは固定名、各実行もバックアップ）
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    model_path_best = out_dir / "policy_model.pt"
    model_path_run  = out_dir / f"policy_model_run_{run_tag}.pt"
    stats_path = out_dir / STATS_TXT

    # init_model を読む（あれば）＋ best_valid_acc1 を引き継ぐ
    best_valid_acc1 = -1.0
    if getattr(cfg, "init_model", ""):
        init_path = Path(cfg.init_model)
        if not init_path.exists():
            raise SystemExit("init_model not found: " + str(init_path))

        ck = torch.load(str(init_path), map_location=device)
        if not (isinstance(ck, dict) and "model_state" in ck):
            raise SystemExit("init_model format unexpected: " + str(init_path))

        model.load_state_dict(ck["model_state"], strict=True)
        best_valid_acc1 = float(ck.get("best_valid_acc1", -1.0))
        print("loaded init model:", str(init_path))
        print("resume best_valid_acc1:", best_valid_acc1)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = amp.GradScaler(enabled=(device.type == "cuda"))

    lines = []
    lines.append(f"device={device}")
    lines.append(f"epochs={cfg.epochs}")
    lines.append(f"batch_size={cfg.batch_size}")
    lines.append(f"lr={cfg.lr}")
    lines.append(f"weight_decay={cfg.weight_decay}")
    lines.append(f"vocab_size={vocab_size}")
    lines.append(f"train_lines={len(train_ds)}")
    lines.append(f"valid_lines={len(valid_ds)}")
    lines.append(f"start_best_valid_acc1={best_valid_acc1}")
    lines.append("")

    for ep in range(1, cfg.epochs + 1):
        print(f"epoch {ep}/{cfg.epochs}")

        tr_loss, tr_acc1, tr_acc5 = train_one_epoch(
            model, train_loader, optimizer, device, scaler, log_every=cfg.log_every
        )
        va_loss, va_acc1, va_acc5 = eval_one_epoch(model, valid_loader, device)

        msg = (
            f"epoch={ep} "
            f"train_loss={tr_loss:.4f} train_acc1={tr_acc1:.4f} train_acc5={tr_acc5:.4f} "
            f"valid_loss={va_loss:.4f} valid_acc1={va_acc1:.4f} valid_acc5={va_acc5:.4f}"
        )
        print(msg)
        lines.append(msg)

        # 改善したときだけ保存（上書き事故防止）
        if va_acc1 > best_valid_acc1:
            best_valid_acc1 = va_acc1
            payload = {
                "model_state": model.state_dict(),
                "vocab_json": str(vocab_json),
                "vocab_size": vocab_size,
                "cfg": cfg.__dict__,
                "best_valid_acc1": best_valid_acc1,
            }
            torch.save(payload, model_path_best)
            torch.save(payload, model_path_run)
            print("saved best model:", str(model_path_best))
            print("saved run backup :", str(model_path_run))
        else:
            print("not improved; skip saving. best_valid_acc1=", best_valid_acc1)

    stats_path.write_text("\n".join(lines), encoding="utf-8")
    print("saved stats:", str(stats_path))

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_vocab = sub.add_parser("build_vocab")
    ap_vocab.add_argument("--train_jsonl", default=POSITIONS_TRAIN)
    ap_vocab.add_argument("--out_vocab", default=str(Path(OUT_DIR) / VOCAB_JSON))
    ap_vocab.add_argument("--max_unique", type=int, default=200000)

    ap_train = sub.add_parser("train")
    ap_train.add_argument("--vocab_json", default=str(Path(OUT_DIR) / VOCAB_JSON))
    ap_train.add_argument("--epochs", type=int, default=3)
    ap_train.add_argument("--batch_size", type=int, default=256)
    ap_train.add_argument("--lr", type=float, default=1e-3)
    ap_train.add_argument("--weight_decay", type=float, default=1e-4)
    ap_train.add_argument("--num_workers", type=int, default=2)
    ap_train.add_argument("--log_every", type=int, default=200)
    ap_train.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap_train.add_argument("--init_model", default="")


    args = ap.parse_args()

    if args.cmd == "build_vocab":
        build_move_vocab(args.train_jsonl, args.out_vocab, max_unique=args.max_unique)
        return

    if args.cmd == "train":
        cfg = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            log_every=args.log_every,
            device=args.device,
            init_model=args.init_model,   # ★追加
        )

        run_train(cfg, args.vocab_json)
        return


if __name__ == "__main__":
    main()
