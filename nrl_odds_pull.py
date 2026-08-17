"""Snapshot NRL betting lines from The Odds API.

Pulls h2h / spreads / totals for AU bookmakers, prints a consensus
(median) line per fixture, and appends every book's prices to
nrl_odds_log.csv so line movement is preserved across runs.

Usage: python nrl_odds_pull.py
Key: THE_ODDS_API_KEY in the root .env.
"""

import csv
import os
from datetime import datetime, timezone
from statistics import median

import requests

ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nrl_odds_log.csv")
SPORT = "rugbyleague_nrl"


def api_key():
    if os.environ.get("THE_ODDS_API_KEY"):
        return os.environ["THE_ODDS_API_KEY"]
    for line in open(ENV):
        if line.startswith("THE_ODDS_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("THE_ODDS_API_KEY not found in env or .env")


r = requests.get(
    f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds",
    params={"apiKey": api_key(), "regions": "au",
            "markets": "h2h,spreads,totals", "oddsFormat": "decimal"},
    timeout=30)
r.raise_for_status()
games = r.json()
print(f"{len(games)} games | quota remaining: {r.headers.get('x-requests-remaining')}\n")

snap_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
new_log_rows = []

for g in games:
    home, away = g["home_team"], g["away_team"]
    spreads, totals, h2h_home, h2h_away = [], [], [], []
    for bk in g["bookmakers"]:
        row = {"snapshot": snap_time, "commence": g["commence_time"],
               "home_team": home, "away_team": away, "book": bk["key"],
               "home_spread": "", "home_spread_price": "", "away_spread_price": "",
               "total": "", "over_price": "", "under_price": "",
               "h2h_home": "", "h2h_away": ""}
        for mk in bk["markets"]:
            oc = {o["name"]: o for o in mk["outcomes"]}
            if mk["key"] == "spreads" and home in oc:
                row["home_spread"] = oc[home].get("point", "")
                row["home_spread_price"] = oc[home]["price"]
                row["away_spread_price"] = oc.get(away, {}).get("price", "")
                if oc[home].get("point") is not None:
                    spreads.append(oc[home]["point"])
            elif mk["key"] == "totals":
                over = oc.get("Over", {})
                row["total"] = over.get("point", "")
                row["over_price"] = over.get("price", "")
                row["under_price"] = oc.get("Under", {}).get("price", "")
                if over.get("point") is not None:
                    totals.append(over["point"])
            elif mk["key"] == "h2h":
                row["h2h_home"] = oc.get(home, {}).get("price", "")
                row["h2h_away"] = oc.get(away, {}).get("price", "")
                if home in oc:
                    h2h_home.append(oc[home]["price"])
                if away in oc:
                    h2h_away.append(oc[away]["price"])
        new_log_rows.append(row)

    line = f"  spread (home) {median(spreads):+.1f}" if spreads else "  no spread"
    tot = f"  total {median(totals):.1f}" if totals else "  no total"
    prices = (f"  h2h {median(h2h_home):.2f}/{median(h2h_away):.2f}"
              if h2h_home and h2h_away else "")
    n_books = len(g["bookmakers"])
    print(f"{home} v {away}  ({g['commence_time'][:16]}, {n_books} books)")
    print(f"{line}{tot}{prices}\n")

fields = list(new_log_rows[0].keys()) if new_log_rows else []
exists = os.path.exists(LOG)
with open(LOG, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    if not exists:
        w.writeheader()
    w.writerows(new_log_rows)
print(f"{len(new_log_rows)} book-lines appended to {LOG}")
