import cshogi

b = cshogi.Board()
print("has legal_moves:", hasattr(b, "legal_moves"))
print("has generate_legal_moves:", hasattr(b, "generate_legal_moves"))
print("has is_legal:", hasattr(b, "is_legal"))
print("sample sfen:", b.sfen())

try:
    lm = b.legal_moves
    # イテレータ/リスト化できるか
    lst = list(lm)
    print("legal_moves count:", len(lst))
    # 先頭数個をUSIに
    print("first moves usi:", [cshogi.move_to_usi(m) for m in lst[:10]])
except Exception as e:
    print("legal_moves access failed:", type(e).__name__, e)

try:
    gen = b.generate_legal_moves()
    lst2 = list(gen)
    print("generate_legal_moves count:", len(lst2))
    print("first moves usi:", [cshogi.move_to_usi(m) for m in lst2[:10]])
except Exception as e:
    print("generate_legal_moves failed:", type(e).__name__, e)
