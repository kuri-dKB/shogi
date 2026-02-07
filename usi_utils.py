# usi_utils.py

RANK_CHARS = "abcdefghi"
FILE_CHARS = "123456789"

def rotate_usi(usi: str) -> str:
    """
    USI文字列を180度回転させる（先手視点⇔後手視点の変換）。
    例: '7g7f' -> '3c3d', 'P*7g' -> 'P*3c'
    """
    # 打ち (Drop): "P*7g"
    if "*" in usi:
        piece, dst = usi.split("*")
        # piece (P, L, ...) は変わらない
        dst_rot = _rotate_sq(dst)
        return f"{piece}*{dst_rot}"
    
    # 移動 (Move): "7g7f", "7g7f+"
    # 末尾の成(+)はそのまま
    promote = ""
    if usi.endswith("+"):
        promote = "+"
        usi = usi[:-1]
    
    src = usi[:2]
    dst = usi[2:]
    
    return f"{_rotate_sq(src)}{_rotate_sq(dst)}{promote}"

def _rotate_sq(sq: str) -> str:
    """座標文字列 '7g' 等を180度回転"""
    if len(sq) != 2:
        return sq # エラー回避あるいはそのまま
    f_char = sq[0] # '7'
    r_char = sq[1] # 'g'
    
    # 筋: 1<->9, 2<->8 ... ('1' is ord 49)
    # 10 - int(f)
    f_new = str(10 - int(f_char))
    
    # 段: a<->i, b<->h ...
    r_idx = RANK_CHARS.index(r_char)
    r_new = RANK_CHARS[8 - r_idx]
    
    return f_new + r_new
