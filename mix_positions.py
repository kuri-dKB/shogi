import shutil
from pathlib import Path

BASE_TRAIN = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\positions_train_s20_e2.jsonl"
SELFPLAY = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\positions_selfplay.jsonl"
OUT_MIX = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\positions_train_mix.jsonl"

MAX_SELFPLAY_LINES = 10000

def main():
    base = Path(BASE_TRAIN)
    sp = Path(SELFPLAY)
    out = Path(OUT_MIX)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not base.exists():
        raise SystemExit("missing base: " + str(base))
    if not sp.exists():
        raise SystemExit("missing selfplay: " + str(sp))

    # base を out にコピー（最初は教師ありだけ）
    shutil.copyfile(base, out)

    # selfplay を最大 MAX_SELFPLAY_LINES 行だけ追記
    appended = 0
    with out.open("a", encoding="utf-8") as f_out, sp.open("r", encoding="utf-8") as f_sp:
        for line in f_sp:
            if appended >= MAX_SELFPLAY_LINES:
                break
            f_out.write(line)
            appended += 1

    print("DONE")
    print("selfplay_lines_appended =", appended)
    print("base_size =", base.stat().st_size)
    print("selfplay_size =", sp.stat().st_size)
    print("mix_size =", out.stat().st_size)
    print("out =", str(out))

if __name__ == "__main__":
    main()
