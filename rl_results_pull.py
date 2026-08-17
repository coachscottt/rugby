"""Pull rugby league results from rugbyleagueproject.org into per-league CSVs.

Usage: python rl_results_pull.py [nrl|sl]   (default: both)

Writes nrl_2026_results.csv / sl_2026_results.csv with columns:
round, date, home_team, away_team, home_score, away_score, venue.
Home team is listed first on RLP results pages.
"""

import csv
import os
import re
import sys
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LEAGUES = {
    "nrl": {
        "url": "https://www.rugbyleagueproject.org/seasons/nrl-2026/results.html",
        "season_slug": "nrl-2026",
        "out": rf"{BASE_DIR}\nrl_2026_results.csv",
        "teams": {
            "newcastle-knights": "Newcastle Knights",
            "north-queensland-cowboys": "North Queensland Cowboys",
            "canterbury-bankstown-bulldogs": "Canterbury Bulldogs",
            "st-george-illawarra-dragons": "St George Illawarra Dragons",
            "melbourne": "Melbourne Storm",
            "parramatta-eels": "Parramatta Eels",
            "warriors": "New Zealand Warriors",
            "sydney-roosters": "Sydney Roosters",
            "brisbane-broncos": "Brisbane Broncos",
            "penrith-panthers": "Penrith Panthers",
            "canberra-raiders": "Canberra Raiders",
            "cronulla-sutherland-sharks": "Cronulla Sharks",
            "cronulla-sharks": "Cronulla Sharks",
            "cronulla": "Cronulla Sharks",
            "gold-coast-titans": "Gold Coast Titans",
            "manly-warringah-sea-eagles": "Manly Sea Eagles",
            "south-sydney-rabbitohs": "South Sydney Rabbitohs",
            "wests-tigers": "Wests Tigers",
            "dolphins": "Dolphins",
        },
    },
    "sl": {
        "url": "https://www.rugbyleagueproject.org/seasons/super-league-2026/results.html",
        "season_slug": "super-league-2026",
        "out": rf"{BASE_DIR}\sl_2026_results.csv",
        "teams": {
            "bradford-bulls": "Bradford Bulls",
            "castleford-tigers": "Castleford Tigers",
            "catalans-dragons": "Catalans Dragons",
            "huddersfield-giants": "Huddersfield Giants",
            "hull-fc": "Hull FC",
            "hull-kingston-rovers": "Hull Kingston Rovers",
            "leeds-rhinos": "Leeds Rhinos",
            "leigh-leopards": "Leigh Leopards",
            "st-helens": "St Helens",
            "toulouse-olympique": "Toulouse Olympique",
            "wakefield-trinity": "Wakefield Trinity",
            "warrington-wolves": "Warrington Wolves",
            "wigan-warriors": "Wigan Warriors",
            "york-knights": "York Knights",
        },
    },
}

MONTHS = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
          "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}


def pull(league):
    cfg = LEAGUES[league]
    req = urllib.request.Request(cfg["url"], headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")

    slug = cfg["season_slug"]
    tokens = re.split(
        rf'<a href="/seasons/{slug}/[^"]*?/summary\.html">(Round \d+|[^<]*Final[^<]*)</a>',
        html)
    row_re = re.compile(
        rf'<td align="right">([^<]*)</td><td>([^<]*)</td>'
        rf'<td(?: class="team")?><a href="/seasons/{slug}/([a-z-]+)/summary\.html">[^<]+</a></td>'
        rf'<td class="n">(\d+)</td>'
        rf'<td(?: class="team")?><a href="/seasons/{slug}/([a-z-]+)/summary\.html">[^<]+</a></td>'
        rf'<td class="n">(\d+)</td>'
        rf'.*?<a href="/venues/\d+">([^<]*)</a>', re.S)

    rows, cur_month = [], None
    for i in range(1, len(tokens), 2):
        rnd, seg = tokens[i], tokens[i + 1]
        if "Final" in rnd:
            continue
        for m in row_re.finditer(seg):
            date_raw, _, hslug, hs, aslug, aws, venue = [x.strip() for x in m.groups()]
            parts = date_raw.split()
            if len(parts) == 2 and parts[0] in MONTHS:
                cur_month, day = MONTHS[parts[0]], int(parts[1])
            elif date_raw.isdigit():
                day = int(date_raw)
            else:
                continue
            home = cfg["teams"].get(hslug, hslug)
            away = cfg["teams"].get(aslug, aslug)
            rows.append([rnd, f"2026-{cur_month:02d}-{day:02d}", home, away,
                         int(hs), int(aws), venue])

    # Keep manually entered results the site has not published yet: merge on
    # (date, home, away) and let the scrape win where both have a row.
    kept = []
    if os.path.exists(cfg["out"]):
        seen = {(r[1], r[2], r[3]) for r in rows}
        with open(cfg["out"], newline="", encoding="utf-8") as f:
            for old in csv.reader(f):
                if len(old) == 7 and old[0] != "round" and (old[1], old[2], old[3]) not in seen:
                    kept.append(old)
    allrows = sorted(rows + kept, key=lambda r: (int(r[0].split()[1]), r[1]))

    with open(cfg["out"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["round", "date", "home_team", "away_team",
                    "home_score", "away_score", "venue"])
        w.writerows(allrows)

    teams = sorted({r[2] for r in allrows} | {r[3] for r in allrows})
    known = set(cfg["teams"].values())
    unmapped = sorted(t for t in teams if t not in known)
    print(f"{league}: {len(allrows)} matches -> {cfg['out']}"
          + (f"  ({len(kept)} kept from manual entry)" if kept else ""))
    print(f"  rounds {allrows[0][0]} - {allrows[-1][0]}, "
          f"last date {max(r[1] for r in allrows)}")
    print(f"  {len(teams)} teams" + (f"  UNMAPPED: {unmapped}" if unmapped else ""))


for lg in (sys.argv[1:] or ["nrl", "sl"]):
    pull(lg)
