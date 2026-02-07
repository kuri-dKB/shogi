import json
from pathlib import Path

import shogi

DATASET_JSONL = r"C:\Users\ug417\.gemini\antigravity\scratch\将棋強化学習\shogidb2_dump\dataset_train.jsonl"


def board_from_csa_initial(initial_lines):
    """
    initial_board に入っている P1..P9 と 手番(単独の+/-) から、
    python-shogi の Board に設定する。

    方針：
    ここではCSAの盤面を一旦「CSA文字列」として組み立てて、
    python-shogi 側のCSA読込機能を使う。
    """
    # python-shogi には CSA のパーサがあるので、それに任せるのが安全
    # 盤面定義の行をまとめて CSA 断片にする
    csa_text = "\n".join(initial_lines) + "\n"
    # CSAの初期局面だけを読み込ませるために、ダミーで一手だけ付ける必要がある場合がある
    # なので、まずは Board() を作って CSA.Parser を使って局面を反映する
    parser = shogi.CSA.Parser.parse_str(csa_text)
    # parse_str は games を返す。初期局面のみの入力だと games が空になる場合があるため対策
    # → 盤面が反映されない場合は別方法に切り替える
    if not parser:
        raise ValueError("CSA initial board parse failed (no games returned)")
    game = parser[0]
    return game.board


def apply_csa_moves(board: shogi.Board, moves):
    """
    moves: ['+7776FU', '-3334FU', ...]
    python-shogi はCSA形式の指し手文字列を move に変換できる。
    """
    for mv in moves:
        # python-shogi は手番を board.turn で管理するので、
        # mv先頭の+/- と board.turn が一致するかを軽くチェックしてズレを早期発見
        side = mv[0]
        expected = '+' if board.turn == shogi.BLACK else '-'
        if side != expected:
            # ここで無理に修正すると事故るのでエラーにする
            raise ValueError(f"turn mismatch: mv={mv} expected={expected}")

        move = shogi.CSA.Parser.parse_move_str(mv, board)
        board.push(move)


def main():
    path = Path(DATASET_JSONL)
    n_ok = 0
    n_fail = 0

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            obj = json.loads(line)
            game_id = obj["game_id"]

            try:
                # 初期局面
                board = board_from_csa_initial(obj["initial_board"])
                # 指し手適用
                apply_csa_moves(board, obj["moves"])
                n_ok += 1

            except Exception as e:
                n_fail += 1
                print(f"[FAIL] {i} game_id={game_id} error={type(e).__name__}: {e}")
                # まず原因調査のため、最初の数件で止める
                if n_fail >= 5:
                    break

            # 最初は少数で十分（負荷も少ない）
            if i >= 200:
                break

    print(f"checked={n_ok+n_fail} ok={n_ok} fail={n_fail}")


if __name__ == "__main__":
    main()
