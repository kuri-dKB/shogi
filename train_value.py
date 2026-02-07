# train_value.py
import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import Dataset, DataLoader


# ====== パス（あなたの環境に合わせて変えてOK） ======
POSITIONS_VALUE_JSONL = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\positions_selfplay_value.jsonl"
OUT_DIR = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\value_model"

MODEL_PT = "value_model.pt"
STATS_TXT = "train_stats.txt"


# ====== SFEN -> 入力テンソル（train_policy と同じ思想） ======
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
PROMOTED_KIND_SHIFT = {
    "P": 8,   # +P
    "L": 9,   # +L
    "N": 10,  # +N
    "S": 11,  # +S
    "B": 12,  # +B
    "R": 13,  # +R
}
HAND_ORDER = ["P", "L", "N", "S", "G", "B", "R"]


def parse_sfen(sfen: str) -> Tuple[str, str, str, str]:
    parts = sfen.strip().split()
    if len(parts) < 4:
        raise ValueError(f"bad sfen: {sfen}")
    return parts[0], parts[1], parts[2], parts[3]


def parse_board_part(board_part: str):
    ranks = board_part.split("/")
    if len(ranks) != 9:
        raise ValueError(f"bad board part: {board_part}")
    return ranks


def piece_token_to_kind14(token: str) -> int:
    promoted = False
    if token.startswith("+"):
        promoted = True
        token = token[1:]

    base = token.upper()
    if base not in PIECE_BASE_TO_KIND:
        raise ValueError(f"unknown piece token: {token}")

    if promoted and base in PROMOTED_KIND_SHIFT:
        return PROMOTED_KIND_SHIFT[base]
    return PIECE_BASE_TO_KIND[base]


def parse_hands(hands_part: str):
    b = {k: 0 for k in HAND_ORDER}
    w = {k: 0 for k in HAND_ORDER}
    if hands_part == "-" or hands_part == "":
        return b, w

    i = 0
    n = len(hands_part)
    while i < n:
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

    turn_is_black = (turn_part == "b")

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

            is_black_piece = ch.isupper()
            mine = (is_black_piece == turn_is_black)

            kind = piece_token_to_kind14(token)
            ch_idx = kind if mine else (14 + kind)

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

    x[28, :, :] = 1.0

    b_hand, w_hand = parse_hands(hands_part)
    my_hand = b_hand if turn_is_black else w_hand
    op_hand = w_hand if turn_is_black else b_hand

    denom = 18.0
    for i, k in enumerate(HAND_ORDER):
        x[29 + i, :, :] = float(my_hand[k]) / denom
        x[36 + i, :, :] = float(op_hand[k]) / denom

    return x


# ====== Dataset（z を 0/1 にする） ======
class ValueDataset(Dataset):
    def __init__(self, jsonl_path: str, offsets):
        self.path = Path(jsonl_path)
        self.offsets = offsets
        print(f"dataset lines={len(self.offsets)} file={self.path}")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int):
        with self.path.open("rb") as f:
            f.seek(self.offsets[idx])
            line = f.readline().decode("utf-8")
        obj = json.loads(line)

        sfen = obj["sfen"]
        z = int(obj["z"])  # -1 or +1

        # 負け=-1 -> 0.0 / 勝ち=+1 -> 1.0
        y = 1.0 if z > 0 else 0.0

        x = sfen_to_tensor_self_view(sfen)
        return torch.from_numpy(x), torch.tensor([y], dtype=torch.float32)


def index_jsonl_offsets(path: str):
    p = Path(path)
    offsets = []
    with p.open("rb") as f:
        pos = 0
        while True:
            line = f.readline()
            if not line:
                break
            offsets.append(pos)
            pos = f.tell()
    return offsets


def split_offsets(offsets, valid_percent: float, seed: int):
    rng = random.Random(seed)
    idx = list(range(len(offsets)))
    rng.shuffle(idx)
    n_valid = max(1, int(len(idx) * valid_percent / 100.0))
    valid_idx = set(idx[:n_valid])
    tr = [offsets[i] for i in range(len(offsets)) if i not in valid_idx]
    va = [offsets[i] for i in range(len(offsets)) if i in valid_idx]
    return tr, va


# ====== Model（value を 1つ出すだけ） ======
class SimpleValueNet(nn.Module):
    def __init__(self):
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
        self.fc = nn.Linear(32 * 9 * 9, 1)

    def forward(self, x):
        h = self.body(x)
        h = self.head(h)
        h = torch.flatten(h, 1)
        return self.fc(h)  # (B,1)


@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    log_every: int = 200
    valid_percent: float = 1.0
    seed: int = 1234
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    init_model: str = ""


def bin_acc(logits: torch.Tensor, y: torch.Tensor) -> float:
    # logits -> sigmoid -> 0/1
    with torch.no_grad():
        p = torch.sigmoid(logits)
        pred = (p >= 0.5).float()
        return (pred == y).float().mean().item()


def train_one_epoch(model, loader, optimizer, device, scaler, log_every=200):
    model.train()
    bce = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_n = 0
    acc_sum = 0.0

    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with amp.autocast(enabled=(device.type == "cuda")):
            logits = model(x)
            loss = bce(logits, y)

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
        acc_sum += bin_acc(logits, y) * bs

        if step % log_every == 0:
            print(f"step={step} loss={total_loss/total_n:.4f} acc={acc_sum/total_n:.4f}")

    return total_loss / total_n, acc_sum / total_n


def eval_one_epoch(model, loader, device):
    model.eval()
    bce = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_n = 0
    acc_sum = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = bce(logits, y)

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_n += bs
            acc_sum += bin_acc(logits, y) * bs

    return total_loss / total_n, acc_sum / total_n


def safe_torch_load(path: str, device):
    # PyTorch 2.1+ では weights_only=True が使える。古い場合は fallback。
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def run_train(cfg: TrainConfig, jsonl_path: str):
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    offsets = index_jsonl_offsets(jsonl_path)
    tr_off, va_off = split_offsets(offsets, valid_percent=cfg.valid_percent, seed=cfg.seed)

    train_ds = ValueDataset(jsonl_path, tr_off)
    valid_ds = ValueDataset(jsonl_path, va_off)

    device = torch.device(cfg.device)

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

    model = SimpleValueNet().to(device)

    best_valid_acc = -1.0
    if cfg.init_model:
        init_path = Path(cfg.init_model)
        if init_path.exists():
            ck = safe_torch_load(str(init_path), device)
            if isinstance(ck, dict) and "model_state" in ck:
                model.load_state_dict(ck["model_state"], strict=True)
                # best_valid_acc = float(ck.get("best_valid_acc", -1.0))
                best_valid_acc = -1.0
                print("loaded init model:", str(init_path))
                print("reset best_valid_acc to -1.0")
            else:
                raise SystemExit("init_model format unexpected: " + str(init_path))
        else:
            raise SystemExit("init_model not found: " + str(init_path))

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = amp.GradScaler(enabled=(device.type == "cuda"))

    stats_path = out_dir / STATS_TXT
    model_path = out_dir / MODEL_PT

    lines = []
    lines.append(f"device={device}")
    lines.append(f"epochs={cfg.epochs}")
    lines.append(f"batch_size={cfg.batch_size}")
    lines.append(f"lr={cfg.lr}")
    lines.append(f"weight_decay={cfg.weight_decay}")
    lines.append(f"train_lines={len(train_ds)}")
    lines.append(f"valid_lines={len(valid_ds)}")
    lines.append("")

    for ep in range(1, cfg.epochs + 1):
        print(f"epoch {ep}/{cfg.epochs}")

        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, scaler, log_every=cfg.log_every)
        va_loss, va_acc = eval_one_epoch(model, valid_loader, device)

        msg = f"epoch={ep} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} valid_loss={va_loss:.4f} valid_acc={va_acc:.4f}"
        print(msg)
        lines.append(msg)

        if va_acc > best_valid_acc:
            best_valid_acc = va_acc
            payload = {
                "model_state": model.state_dict(),
                "cfg": cfg.__dict__,
                "best_valid_acc": best_valid_acc,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            torch.save(payload, model_path)
            print("saved best model:", str(model_path))
        else:
            print("not improved; skip saving. best_valid_acc=", best_valid_acc)

    stats_path.write_text("\n".join(lines), encoding="utf-8")
    print("saved stats:", str(stats_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=POSITIONS_VALUE_JSONL)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--valid_percent", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--init_model", default="")

    args = ap.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        log_every=args.log_every,
        valid_percent=args.valid_percent,
        seed=args.seed,
        device=args.device,
        init_model=args.init_model,
    )
    run_train(cfg, args.jsonl)


if __name__ == "__main__":
    main()
