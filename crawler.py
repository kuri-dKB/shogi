import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


BASE = "https://shogidb2.com"

TOURNAMENT_URLS: Dict[str, str] = {
    "順位戦": "https://shogidb2.com/tournament/%E9%A0%86%E4%BD%8D%E6%88%A6",
    "竜王戦": "https://shogidb2.com/tournament/%E7%AB%9C%E7%8E%8B%E6%88%A6",
    "名人戦": "https://shogidb2.com/tournament/%E5%90%8D%E4%BA%BA%E6%88%A6",
    "王位戦": "https://shogidb2.com/tournament/%E7%8E%8B%E4%BD%8D%E6%88%A6",
    "王座戦": "https://shogidb2.com/tournament/%E7%8E%8B%E5%BA%A7%E6%88%A6",
    "棋王戦": "https://shogidb2.com/tournament/%E6%A3%8B%E7%8E%8B%E6%88%A6",
    "棋聖戦": "https://shogidb2.com/tournament/%E6%A3%8B%E8%81%96%E6%88%A6",
    "王将戦": "https://shogidb2.com/tournament/%E7%8E%8B%E5%B0%86%E6%88%A6",
}

OUTPUT_DIR = "shogidb2_dump"
GAMES_LIST_DIR = os.path.join(OUTPUT_DIR, "game_lists")
GAMES_JSON_DIR = os.path.join(OUTPUT_DIR, "games_json")

# 負荷を抑えるための待機（秒）
SLEEP_BASE = 1.5          # 基本待機
SLEEP_JITTER = 0.7        # 乱数でブレを入れる（アクセスが機械的にならないように）
TIMEOUT = 20              # 通信タイムアウト
MAX_RETRIES = 5           # 再試行回数


@dataclass
class FetchResult:
    url: str
    status: int
    text: str


def polite_sleep(multiplier: float = 1.0) -> None:
    t = (SLEEP_BASE + random.random() * SLEEP_JITTER) * multiplier
    time.sleep(t)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ja,en;q=0.8",
        }
    )
    return s


def fetch_html(session: requests.Session, url: str) -> FetchResult:
    # 失敗しても「止めずに返す」ため、ここでは例外を投げない方針にする
    backoff = 1.0
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=TIMEOUT)
            # 普通に本文は返す（404でも本文があることがある）
            return FetchResult(url=url, status=r.status_code, text=r.text)
        except Exception as e:
            last_exc = e
            polite_sleep(multiplier=backoff)
            backoff *= 2.0
    # ここまで来たら通信自体がダメ
    return FetchResult(url=url, status=0, text=f"__FETCH_FAILED__ {last_exc}")


def normalize_game_id(game_url: str) -> str:
    # /games/<hex> を取り出してファイル名に使う
    p = urlparse(game_url).path
    m = re.search(r"/games/([0-9a-f]{20,})", p)
    if not m:
        # 想定外URLでも一意に
        m = re.search(r"/games/([^/?#]+)", p)
        if not m:
            return re.sub(r"[^a-zA-Z0-9]+", "_", p).strip("_")
    return m.group(1)


def parse_tournament_page(html: str, current_url: str) -> Tuple[List[str], Optional[str]]:
    soup = BeautifulSoup(html, "html.parser")
    game_urls: List[str] = []
    for a in soup.select("a"):
        href = a.get("href")
        if not href:
            continue
        if "/games/" in href:
            abs_url = urljoin(current_url, href)
            game_urls.append(abs_url)

    next_url = None
    next_a = soup.find('a', string=re.compile(r"次|>|Next"))
    if next_a and next_a.get("href"):
        next_url = urljoin(current_url, next_a.get("href"))
    else:
        curr_page_m = re.search(r"page=(\d+)", current_url)
        curr_page = int(curr_page_m.group(1)) if curr_page_m else 1
        target_page = curr_page + 1
        for a in soup.select("a"):
            href = a.get("href")
            if href and f"page={target_page}" in href:
                next_url = urljoin(current_url, href)
                break

    seen = set()
    uniq = []
    for u in game_urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq, next_url


def crawl_tournament(session: requests.Session, name: str, start_url: str) -> List[str]:
    os.makedirs(GAMES_LIST_DIR, exist_ok=True)
    out_path = os.path.join(GAMES_LIST_DIR, f"{name}.txt")
    existing = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.add(line)
    collected: List[str] = []
    url = start_url
    page = 1
    # テスト時は1ページだけで止まるようにすることも可能だが、全ページ拾う
    while url:
        print(f"[{name}] page {page} -> {url}")
        polite_sleep()
        res = fetch_html(session, url)
        if res.status != 200:
            break
        game_urls, next_url = parse_tournament_page(res.text, url)
        new_count = 0
        with open(out_path, "a", encoding="utf-8") as f:
            for gu in game_urls:
                if gu in existing:
                    continue
                f.write(gu + "\n")
                existing.add(gu)
                collected.append(gu)
                new_count += 1
        print(f"[{name}] +{new_count} games")
        url = next_url
        page += 1
        # 全量クロールを避けるため、テスト時はここで止める設定も可能
        # if page > 1: break 
    return sorted(existing)


def _extract_var_data_style(html: str) -> Optional[Dict]:
    idx = html.find("var data")
    if idx == -1:
        return None
    sub = html[idx:]
    brace_start = sub.find("{")
    if brace_start == -1:
        return None
    i = brace_start
    depth = 0
    in_str = False
    esc = False
    for j in range(i, len(sub)):
        ch = sub[j]
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        else:
            if ch == '"': in_str = True; continue
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    obj_text = sub[i : j + 1]
                    try: return json.loads(obj_text)
                    except Exception: return None
    return None


def _extract_next_data(html: str) -> Optional[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    s = soup.find("script", attrs={"id": "__NEXT_DATA__"})
    if not s or not s.string:
        return None
    try: return json.loads(s.string)
    except Exception: return None


def _extract_nuxt_state(html: str) -> Optional[Dict]:
    m = re.search(r"window\.__NUXT__\s*=\s*(\{.*?\});", html, flags=re.DOTALL)
    if not m:
        return None
    txt = m.group(1)
    try: return json.loads(txt)
    except Exception: return None


def _extract_any_json_like_script(html: str) -> Optional[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.find_all("script")
    for sc in scripts:
        if not sc.string: continue
        t = sc.string
        if ("moves" not in t) and ("kifu" not in t) and ("sfen" not in t) and ("hands" not in t) and ("game" not in t):
            continue
        m = re.search(r"(\{.*\})", t, flags=re.DOTALL)
        if not m: continue
        cand = m.group(1).strip()
        if len(cand) < 50: continue
        try: return json.loads(cand)
        except Exception: continue
    return None


def extract_game_data(html: str) -> Tuple[Optional[Dict], str]:
    d = _extract_var_data_style(html)
    if d is not None: return d, "var_data"
    d = _extract_next_data(html)
    if d is not None: return d, "__NEXT_DATA__"
    d = _extract_nuxt_state(html)
    if d is not None: return d, "__NUXT__"
    d = _extract_any_json_like_script(html)
    if d is not None: return d, "generic_script_json"
    return None, "not_found"


def download_games_json(session: requests.Session, game_urls: Iterable[str]) -> None:
    os.makedirs(GAMES_JSON_DIR, exist_ok=True)
    for n, url in enumerate(game_urls, start=1):
        gid = normalize_game_id(url)
        out_path = os.path.join(GAMES_JSON_DIR, f"{gid}.json")
        if os.path.exists(out_path):
            continue
        print(f"[game {n}] {url}")
        polite_sleep()
        res = fetch_html(session, url)
        print(f"  status={res.status}")
        if res.status == 0:
            print("  skip (network error)")
            continue
        if res.status in (404, 410):
            print("  skip (not found)")
            continue
        if res.status != 200:
            print(f"  skip (http {res.status})")
            continue
        data, how = extract_game_data(res.text)
        if data is None:
            print(f"  skip (cannot extract data): {how}")
            continue
        payload = {
            "source_url": url,
            "fetched_at_unix": int(time.time()),
            "extract_method": how,
            "data": data,
        }
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_path, out_path)
        print(f"  saved ({how}) -> {out_path}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session = make_session()
    
    # テスト用：全棋戦を回さず、まずは「順位戦」のリストがあるか確認
    # リストがなければクロールするが、テストなので少なめに
    all_game_urls: List[str] = []
    
    # 既存のリストがあればそれを使う
    tournament_name = "順位戦"
    list_path = os.path.join(GAMES_LIST_DIR, f"{tournament_name}.txt")
    
    if not os.path.exists(list_path):
        print(f"List for {tournament_name} not found, crawling first page...")
        # クロールを1ページだけに制限するフラグを暫定で持たないので、普通に呼ぶ（が、このスクリプト自体の性質上、リスト作成は早い）
        urls = crawl_tournament(session, tournament_name, TOURNAMENT_URLS[tournament_name])
        all_game_urls.extend(urls)
    else:
        with open(list_path, "r", encoding="utf-8") as f:
            for line in f:
                u = line.strip()
                if u: all_game_urls.append(u)

    # 重複排除
    uniq = []
    seen = set()
    for u in all_game_urls:
        if u not in seen:
            seen.add(u); uniq.append(u)

    print(f"Total unique games in list: {len(uniq)}")
    
    # テスト実行：最初の10件だけ
    test_limit = 10
    print(f"--- TEST RUN: first {test_limit} games ---")
    download_games_json(session, uniq[:test_limit])
    
    print("Test run done.")


if __name__ == "__main__":
    main()
