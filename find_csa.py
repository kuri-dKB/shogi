import os
import re
import random
import time
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.common.exceptions import (
    TimeoutException,
    WebDriverException,
    NoSuchElementException,
    StaleElementReferenceException,
    ElementClickInterceptedException,
)

BASE_DIR = Path("shogidb2_dump")
GAME_LISTS_DIR = BASE_DIR / "game_lists"
OUT_DIR = BASE_DIR / "csa_out"
STATE_DIR = BASE_DIR / "state"
DB_PATH = STATE_DIR / "progress.sqlite3"

TARGET_TOURNAMENT_FILES = [
    "順位戦.txt",
    "竜王戦.txt",
    "名人戦.txt",
    "王位戦.txt",
    "王座戦.txt",
    "棋王戦.txt",
    "棋聖戦.txt",
    "王将戦.txt",
]

PAGE_LOAD_TIMEOUT = 25
WAIT_TIMEOUT = 25

SLEEP_BETWEEN_GAMES_MIN = 2.0
SLEEP_BETWEEN_GAMES_MAX = 4.0

SLEEP_AFTER_CLICK_MIN = 0.8
SLEEP_AFTER_CLICK_MAX = 1.8

BACKOFFS = [10, 30, 90]
RESTART_EVERY_N_GAMES = 200


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_game_id(url: str) -> str:
    m = re.search(r"/games/([0-9a-f]{20,})", url)
    if m:
        return m.group(1)
    return re.sub(r"[^a-zA-Z0-9]+", "_", url).strip("_")


def init_db() -> None:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            updated_at INTEGER NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_games_status ON games(status)")
    con.commit()
    con.close()


def upsert_game(game_id: str, url: str) -> None:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    now = int(time.time())
    cur.execute(
        """
        INSERT INTO games(game_id, url, status, attempts, last_error, updated_at)
        VALUES(?, ?, 'pending', 0, NULL, ?)
        ON CONFLICT(game_id) DO UPDATE SET url=excluded.url
        """,
        (game_id, url, now),
    )
    con.commit()
    con.close()


def mark_done(game_id: str) -> None:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    now = int(time.time())
    cur.execute(
        "UPDATE games SET status='done', last_error=NULL, updated_at=? WHERE game_id=?",
        (now, game_id),
    )
    con.commit()
    con.close()


def mark_failed(game_id: str, attempts: int, err: str) -> None:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    now = int(time.time())
    cur.execute(
        """
        UPDATE games
        SET status='failed', attempts=?, last_error=?, updated_at=?
        WHERE game_id=?
        """,
        (attempts, err[:800], now, game_id),
    )
    con.commit()
    con.close()


def set_attempts(game_id: str, attempts: int, err: str) -> None:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    now = int(time.time())
    cur.execute(
        """
        UPDATE games
        SET attempts=?, last_error=?, updated_at=?
        WHERE game_id=?
        """,
        (attempts, err[:800], now, game_id),
    )
    con.commit()
    con.close()


def get_next_batch(limit: int = 300) -> List[Tuple[str, str, int]]:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        SELECT game_id, url, attempts
        FROM games
        WHERE status='pending'
        ORDER BY updated_at ASC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    con.close()
    return rows


def load_urls_from_lists() -> int:
    count = 0
    for fname in TARGET_TOURNAMENT_FILES:
        path = GAME_LISTS_DIR / fname
        if not path.exists():
            print(f"WARNING: list file not found: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                url = line.strip()
                if not url:
                    continue
                gid = normalize_game_id(url)
                upsert_game(gid, url)
                count += 1
    return count


def rand_sleep(a: float, b: float) -> None:
    time.sleep(a + random.random() * (b - a))


def make_driver(headless: bool = False) -> webdriver.Chrome:
    options = ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1280,900")
    options.add_argument("--lang=ja-JP")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)

    try:
        driver.execute_cdp_cmd("Network.enable", {})
        driver.execute_cdp_cmd(
            "Network.setBlockedURLs",
            {
                "urls": [
                    "*doubleclick.net*",
                    "*googlesyndication.com*",
                    "*googleads.g.doubleclick.net*",
                    "*pagead2.googlesyndication.com*",
                    "*googletagservices.com*",
                    "*adservice.google.com*",
                    "*aswift*",
                ]
            },
        )
    except Exception:
        pass

    return driver


def hide_ad_iframes(driver: webdriver.Chrome) -> None:
    js = """
    const iframes = Array.from(document.querySelectorAll('iframe'));
    for (const f of iframes) {
      const id = (f.id || '');
      const src = (f.src || '');
      if (id.startsWith('aswift') || src.includes('doubleclick') || src.includes('googlesyndication') || src.includes('googleads')) {
        f.style.display = 'none';
        f.style.visibility = 'hidden';
        f.style.pointerEvents = 'none';
      }
    }
    """
    try:
        driver.execute_script(js)
    except Exception:
        pass


def looks_like_csa(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    if "V2." not in t[:80]:
        return False
    if re.search(r"^[\+\-]\d{4}[A-Z]{2}", t, flags=re.MULTILINE) is None:
        return False
    if len(t) < 200:
        return False
    return True


def save_csa(game_id: str, csa_text: str) -> Path:
    out_path = OUT_DIR / f"{game_id}.csa"
    tmp_path = OUT_DIR / f"{game_id}.csa.tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(csa_text)
    os.replace(tmp_path, out_path)
    return out_path


def click_element_safely(driver: webdriver.Chrome, element) -> None:
    try:
        element.click()
        return
    except (ElementClickInterceptedException, WebDriverException):
        pass
    try:
        driver.execute_script("arguments[0].click();", element)
    except Exception:
        raise


def find_csa_button(driver: webdriver.Chrome, wait: WebDriverWait):
    xpaths = [
        "//*[self::button or self::a][normalize-space()='CSA形式']",
        "//*[@phx-click='csa']",
        "//*[self::button or self::a][contains(., 'CSA')]",
    ]
    last_err = None
    for xp in xpaths:
        try:
            el = wait.until(EC.presence_of_element_located((By.XPATH, xp)))
            return el
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"CSA button not found: {repr(last_err)}")


def extract_csa_from_page(driver: webdriver.Chrome, url: str) -> str:
    driver.get(url)
    hide_ad_iframes(driver)

    wait = WebDriverWait(driver, WAIT_TIMEOUT)

    btn = find_csa_button(driver, wait)
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", btn)
    time.sleep(0.2)
    hide_ad_iframes(driver)

    click_element_safely(driver, btn)

    rand_sleep(SLEEP_AFTER_CLICK_MIN, SLEEP_AFTER_CLICK_MAX)

    def get_visible_textarea_value() -> Optional[str]:
        try:
            tas = driver.find_elements(By.XPATH, "//textarea")
        except Exception:
            return None
        visible = []
        for ta in tas:
            try:
                if ta.is_displayed():
                    visible.append(ta)
            except Exception:
                continue
        if not visible:
            return None
        for ta in reversed(visible):
            try:
                v = ta.get_attribute("value") or ""
                if v.strip():
                    return v
            except StaleElementReferenceException:
                continue
        try:
            return visible[-1].get_attribute("value") or ""
        except Exception:
            return None

    def csa_ready(_driver):
        hide_ad_iframes(driver)
        v = get_visible_textarea_value()
        if not v:
            return False
        if "V2." in v and len(v) > 200:
            return v
        return False

    value = wait.until(csa_ready)

    return value


def restart_driver(driver: Optional[webdriver.Chrome], headless: bool = False) -> webdriver.Chrome:
    try:
        if driver is not None:
            driver.quit()
    except Exception:
        pass
    time.sleep(3)
    return make_driver(headless=headless)


def main() -> None:
    ensure_dirs()
    init_db()

    inserted = load_urls_from_lists()
    print(f"Loaded/Updated {inserted} URLs in DB.")
    print("Initializing driver...")

    driver = None
    driver = restart_driver(driver, headless=False)

    processed_since_restart = 0

    try:
        while True:
            batch = get_next_batch(limit=300)
            if not batch:
                print("No pending games. Done.")
                break

            for idx, (game_id, url, attempts) in enumerate(batch):
                out_path = OUT_DIR / f"{game_id}.csa"
                if out_path.exists():
                    mark_done(game_id)
                    continue

                if attempts >= len(BACKOFFS):
                    mark_failed(game_id, attempts, "retry limit reached")
                    continue

                print(f"[{idx}] Processing: {url}")

                try:
                    csa = extract_csa_from_page(driver, url)

                    if not looks_like_csa(csa):
                        preview = (csa[:120] if csa else "")
                        raise RuntimeError(f"Extracted text does not match CSA format. preview={preview!r}")

                    saved_path = save_csa(game_id, csa)
                    mark_done(game_id)
                    print(f"  Saved: {saved_path}")

                    rand_sleep(SLEEP_BETWEEN_GAMES_MIN, SLEEP_BETWEEN_GAMES_MAX)

                except TimeoutException as e:
                    attempts += 1
                    err = f"{type(e).__name__}: {repr(e)}"
                    set_attempts(game_id, attempts, err)
                    backoff = BACKOFFS[attempts - 1]
                    print(f"  Timeout. Attempts incremented to {attempts}. Backoff {backoff}s. {err}")
                    time.sleep(backoff)

                except (WebDriverException, ElementClickInterceptedException, StaleElementReferenceException) as e:
                    attempts += 1
                    err = f"{type(e).__name__}: {repr(e)}"
                    set_attempts(game_id, attempts, err)
                    backoff = BACKOFFS[attempts - 1]
                    print(f"  WebDriver error. Attempts incremented to {attempts}. Restarting browser. Backoff {backoff}s. {err}")
                    time.sleep(backoff)
                    driver = restart_driver(driver, headless=False)
                    processed_since_restart = 0

                except Exception as e:
                    attempts += 1
                    err = f"{type(e).__name__}: {repr(e)}"
                    set_attempts(game_id, attempts, err)
                    backoff = BACKOFFS[attempts - 1]
                    print(f"  Error. Attempts incremented to {attempts}. Backoff {backoff}s. {err}")
                    time.sleep(backoff)

                processed_since_restart += 1
                if processed_since_restart >= RESTART_EVERY_N_GAMES:
                    print("Restarting browser for stability...")
                    driver = restart_driver(driver, headless=False)
                    processed_since_restart = 0

    finally:
        try:
            if driver is not None:
                driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
