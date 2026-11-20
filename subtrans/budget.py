"""Daily spend guard.

Every LLM response reports its token usage; we turn it into an estimated cost with a price
table, persist it per UTC day in the cache DB, and refuse new requests once the daily budget
is reached. Requests already in flight can still land, so the budget may be exceeded by at
most one batch per worker.
"""
import sqlite3
import threading
import time

# USD per 1M tokens (input, output). Thinking tokens are billed as output.
# Defaults from the public price lists (Nov 2026); override with PRICES="model=in/out,...".
DEFAULT_PRICES = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.8-flash": (0.75, 3.75),   # introductory price until 2026-12-31, then 1.50 / 7.50
    "gemini-3.7-flash": (0.75, 3.75),
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-3.5-flash": (1.50, 9.00),
}
UNKNOWN_MODEL_PRICE = (1.50, 9.00)  # conservative: an unknown model is priced like an expensive one

_local = threading.local()
_lock = threading.Lock()


def parse_prices(spec):
    prices = dict(DEFAULT_PRICES)
    for item in (spec or "").split(","):
        if "=" not in item:
            continue
        model, rates = item.split("=", 1)
        i, o = rates.split("/")
        prices[model.strip()] = (float(i), float(o))
    return prices


def _conn(path):
    conns = getattr(_local, "conns", None)
    if conns is None:
        conns = _local.conns = {}
    if path not in conns:
        c = sqlite3.connect(path, timeout=30)
        c.execute("CREATE TABLE IF NOT EXISTS usage (day TEXT, model TEXT, input INTEGER, output INTEGER, "
                  "usd REAL, PRIMARY KEY (day, model))")
        c.commit()
        conns[path] = c
    return conns[path]


def today():
    return time.strftime("%Y-%m-%d", time.gmtime())


def cost(model, inp, out, prices):
    pin, pout = prices.get(model, UNKNOWN_MODEL_PRICE)
    return (inp * pin + out * pout) / 1_000_000


def record(path, model, inp, out, prices):
    usd = cost(model, inp, out, prices)
    with _lock:
        c = _conn(path)
        c.execute("INSERT INTO usage VALUES (?,?,?,?,?) ON CONFLICT(day, model) DO UPDATE SET "
                  "input=input+excluded.input, output=output+excluded.output, usd=usd+excluded.usd",
                  (today(), model, inp, out, usd))
        c.commit()
    return usd


def spent_today(path):
    row = _conn(path).execute("SELECT COALESCE(SUM(usd), 0) FROM usage WHERE day=?", (today(),)).fetchone()
    return row[0]


def summary(path, days=7):
    return _conn(path).execute(
        "SELECT day, model, input, output, usd FROM usage ORDER BY day DESC, usd DESC LIMIT ?",
        (days * 10,)).fetchall()
