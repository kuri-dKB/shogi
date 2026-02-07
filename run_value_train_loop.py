import os
import sys
import subprocess
import re
import glob
import time
from pathlib import Path

# Paths
BASE_DIR = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習"
SELFPLAY_SCRIPT = os.path.join(BASE_DIR, "generate_selfplay.py")
MAKE_POS_SCRIPT = os.path.join(BASE_DIR, "make_selfplay_positions.py")
TRAIN_SCRIPT = os.path.join(BASE_DIR, "train_value.py")

SELFPLAY_DATA_DIR = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\selfplay_games"
VALUE_MODEL_DIR = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\value_model"
VALUE_MODEL_PT = os.path.join(VALUE_MODEL_DIR, "value_model.pt")

def run_command(cmd, cwd=BASE_DIR):
    print(f"Running: {cmd}")
    # run and capture output
    result = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error running command:")
        print(result.stdout) # Sometimes error info is in stdout
        print(result.stderr)
        sys.exit(1)
    return result.stdout

def parse_selfplay_log(output):
    # reasons: {'game_over': 500}
    # avg_plies: 105.3
    reasons = re.search(r"reasons:\s*({.*?})", output)
    avg_plies = re.search(r"avg_plies:\s*([\d\.]+)", output)
    return {
        "reasons": reasons.group(1) if reasons else "N/A",
        "avg_plies": avg_plies.group(1) if avg_plies else "N/A"
    }

def parse_make_pos_log(output):
    # winner_majority_ratio = 0.52
    # z_majority_ratio = 0.51
    w_maj = re.search(r"winner_majority_ratio\s*=\s*([\d\.]+)", output)
    z_maj = re.search(r"z_majority_ratio\s*=\s*([\d\.]+)", output)
    return {
        "winner_majority_ratio": w_maj.group(1) if w_maj else "N/A",
        "z_majority_ratio": z_maj.group(1) if z_maj else "N/A"
    }

def parse_train_log(output):
    # epoch=1 train_loss=... train_acc=... valid_loss=0.6900 valid_acc=0.5100
    # We want the last epoch line
    lines = output.strip().split('\n')
    last_log = None
    for line in lines:
        if "valid_loss=" in line:
            last_log = line
    
    if last_log:
        v_loss = re.search(r"valid_loss=([\d\.]+)", last_log)
        v_acc = re.search(r"valid_acc=([\d\.]+)", last_log)
        return {
            "valid_loss": v_loss.group(1) if v_loss else "N/A",
            "valid_acc": v_acc.group(1) if v_acc else "N/A"
        }
    return {"valid_loss": "N/A", "valid_acc": "N/A"}

def clean_selfplay_data():
    files = glob.glob(os.path.join(SELFPLAY_DATA_DIR, "*.json"))
    print(f"Cleaning {len(files)} selfplay files...")
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Failed to remove {f}: {e}")

def main():
    LOOPS = 5
    
    for i in range(1, LOOPS + 1):
        print(f"\n{'='*20} LOOP {i}/{LOOPS} {'='*20}")
        
        # 1. Clean selfplay data
        clean_selfplay_data()
        
        # 2. Selfplay
        print("[Step 1] Generating selfplay games...")
        t0 = time.time()
        out_sp = run_command(f"python \"{SELFPLAY_SCRIPT}\"")
        dt = time.time() - t0
        log_sp = parse_selfplay_log(out_sp)
        print(f"  Done in {dt:.1f}s")
        print(f"  -> Avg Plies: {log_sp['avg_plies']}")
        print(f"  -> Reasons: {log_sp['reasons']}")
        
        # 3. Make positions
        print("[Step 2] Making positions...")
        out_mp = run_command(f"python \"{MAKE_POS_SCRIPT}\"")
        log_mp = parse_make_pos_log(out_mp)
        print(f"  -> Winner Maj Ratio: {log_mp['winner_majority_ratio']} (Warning if > 0.6)")
        print(f"  -> Z Maj Ratio: {log_mp['z_majority_ratio']} (Warning if > 0.55)")
        
        # 4. Train Value
        print("[Step 3] Training value model...")
        cmd_train = f"python \"{TRAIN_SCRIPT}\" --epochs 1"
        if os.path.exists(VALUE_MODEL_PT):
            print(f"  (Loading init model: {VALUE_MODEL_PT})")
            cmd_train += f" --init_model \"{VALUE_MODEL_PT}\""
        
        out_tr = run_command(cmd_train)
        log_tr = parse_train_log(out_tr)
        print(f"  -> Valid Loss: {log_tr['valid_loss']} (Goal: decrease from 0.69)")
        print(f"  -> Valid Acc: {log_tr['valid_acc']} (Goal: 0.5 -> 0.6 -> 0.7)")

if __name__ == "__main__":
    main()
