# utils/input_utils.py
from .logging_utils import YELLOW, RED, GREEN, RESET
import random

def print_progress(step, total_steps=5, prefix=""):
    percent = step / total_steps * 100
    bar_len = 20
    filled = int(bar_len * step // total_steps)
    bar = f"{GREEN}{'█' * filled}{RESET}{'.' * (bar_len - filled)}"
    print(f"\r{prefix}{bar} {YELLOW}{percent:3.0f}%{RESET}", end="", flush=True)

def ask_int(prompt, min_v, max_v):
    while True:
        try:
            v = int(input(f"{YELLOW}{prompt}{RESET}"))
            if min_v <= v <= max_v: return v
            print(f"{RED}От {min_v} до {max_v}.{RESET}")
        except: print(f"{RED}Целое число!{RESET}")

def ask_float(prompt, min_v, max_v):
    while True:
        try:
            v = float(input(f"{YELLOW}{prompt}{RESET}"))
            if min_v <= v <= max_v: return v
            print(f"{RED}От {min_v} до {max_v}.{RESET}")
        except: print(f"{RED}Число!{RESET}")

def parse_int_list(text, min_v, max_v):
    """
    Поддерживает:
    '3,5,7'
    '2-5'
    'all'
    'no limit'
    'r' / '0' -> случайное значение
    'r3' -> 3 случайных значения
    """

    text = text.strip().lower()

    if text in ("no limit", "nolimit", "∞", "inf", "unlimited"):
        return None

    if text == "all":
        return list(range(min_v, max_v + 1))

    result = set()

    parts = [p.strip() for p in text.split(",") if p.strip()]

    for part in parts:

        # r / random / 0
        if part in ("r", "random"):
            result.add(random.randint(min_v, max_v))
            continue

        # r5
        if part.startswith("r") and part[1:].isdigit():
            count = int(part[1:])
            values = list(range(min_v, max_v + 1))
            random.shuffle(values)

            for v in values[:count]:
                result.add(v)

            continue

        # диапазон
        if "-" in part:
            try:
                start, end = map(int, part.split("-"))

                for i in range(start, end + 1):
                    if min_v <= i <= max_v:
                        result.add(i)

            except:
                pass

        # обычное число
        else:
            try:
                val = int(part)

                if min_v <= val <= max_v:
                    result.add(val)

            except:
                pass

    return sorted(result)

def parse_mode_list(text, min_v, max_v):
    """
    Поддерживает:
      - 'all' -> [1,2,...,max_v]
      - '3' -> [3]
      - '2-5' -> [2,3,4,5]
      - 'r' -> случайный эффект
      - 'r3' -> 3 случайных эффекта подряд (один вариант, tuple)
      - 'r-r-r' -> 3 случайных эффекта подряд (один вариант, tuple)
      - '9+12' -> tuple из 9 и 12
      - '9+12,3,5-6' -> смешанные варианты
    """
    text = (text or "").strip().lower()
    if not text:
        return []

    if text == "all":
        return list(range(min_v, max_v + 1))

    def random_tuple(count):
        count = max(1, min(count, max_v - min_v + 1))
        vals = list(range(min_v, max_v + 1))
        random.shuffle(vals)
        return tuple(vals[:count])

    out = []
    parts = [p.strip() for p in text.split(",") if p.strip()]

    for part in parts:
        # r / random / 0
        if part in ("r", "random", "0"):
            out.append(random.randint(min_v, max_v))
            continue

        # r3
        if part.startswith("r") and part[1:].isdigit():
            out.append(random_tuple(int(part[1:])))
            continue

        # r-r-r-r или r+r+r
        if ("r" in part) and all(ch in "r+-" for ch in part):
            tokens = [t for t in part.replace("+", "-").split("-") if t]
            if tokens and all(t == "r" for t in tokens):
                out.append(random_tuple(len(tokens)))
                continue

        # a+b+c
        if "+" in part:
            sub = []
            ok = True
            for token in part.split("+"):
                token = token.strip()
                if not token:
                    ok = False
                    break

                if "-" in token:
                    try:
                        a, b = map(int, token.split("-", 1))
                        for v in range(a, b + 1):
                            if min_v <= v <= max_v:
                                sub.append(v)
                    except:
                        ok = False
                        break
                else:
                    try:
                        v = int(token)
                        if min_v <= v <= max_v:
                            sub.append(v)
                        else:
                            ok = False
                            break
                    except:
                        ok = False
                        break

            if ok and sub:
                out.append(tuple(sub))
            continue

        # диапазон a-b
        if "-" in part:
            try:
                a, b = map(int, part.split("-", 1))
                for v in range(a, b + 1):
                    if min_v <= v <= max_v:
                        out.append(v)
            except:
                continue
            continue

        # одно число
        try:
            v = int(part)
            if min_v <= v <= max_v:
                out.append(v)
        except:
            continue

    return out