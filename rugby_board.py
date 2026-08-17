"""Build the rugby league fair-value board (NRL + Super League).

Reads the model params, round fairs, odds logs, results and the paper-bet
ledger, then writes a self-contained HTML board:

    python rugby_board.py            -> rugby_board.html

Mirrors the football scanner's board: stat tiles, per-league slate tables
with fair v market, open positions with running CLV, and a settled ledger.
All figures are paper only.
"""

import csv
import json
import math
import os
import re
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "rugby_board.html")

LEAGUES = {
    "nrl": {"title": "NRL", "odds": "nrl_odds_log.csv",
            "results": "nrl_2026_results.csv",
            "params": "nrl_model_params.json", "tz": "Australia/Sydney"},
    "sl": {"title": "Super League", "odds": "sl_odds_log.csv",
           "results": "sl_2026_results.csv",
           "params": "sl_model_params.json", "tz": "Europe/London"},
}


def newest_fairs(lg):
    """Highest-numbered {lg}_r<N>_fairs.csv, so a new round is picked up
    automatically once its fairs are logged."""
    best = None
    for name in os.listdir(BASE):
        m = re.fullmatch(rf"{lg}_r(\d+)_fairs\.csv", name)
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), name)
    if best is None:
        raise SystemExit(f"no fairs file found for {lg}")
    return best[1], f"Round {best[0]}"


for _lg, _cfg in LEAGUES.items():
    _cfg["fairs"], _cfg["round"] = newest_fairs(_lg)

# Odds-log team names differ from model/results names in a few places.
ALIAS = {
    "Manly Warringah Sea Eagles": "Manly Sea Eagles",
    "Cronulla Sutherland Sharks": "Cronulla Sharks",
    "Canterbury-Bankstown Bulldogs": "Canterbury Bulldogs",
    "Redcliffe Dolphins": "Dolphins",
    "Hull Kingston Rovers": "Hull Kingston Rovers",
}


def norm(name):
    return ALIAS.get(name.strip(), name.strip())


def read_csv(path):
    with open(os.path.join(BASE, path), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _t_cdf(x, df, scale):
    """Student-t CDF via the regularised incomplete beta (no scipy needed)."""
    t = x / scale
    p = 0.5 * _betainc(df / 2, 0.5, df / (df + t * t))
    return p if t <= 0 else 1 - p


def _betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b) by continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1 - c * d) < 1e-10:
            break
    result = front * (f - 1)
    return result if x < (a + 1) / (a + b + 2) else 1 - _betainc(b, a, 1 - x)


def win_probs(mov, err):
    """Home/draw/away probability from a fair margin and its error model.

    Rugby league margins have fatter tails than a normal in at least one of
    the two leagues, so the curve is whichever of normal / Student-t fitted
    the walk-forward residuals better (see rugby_diagnostics.py). Either way
    it is centred on the fair margin, not on the fitted location.
    """
    if err and err.get("dist") == "t":
        df, scale = err["df"], err["scale"]
        p_away = _t_cdf(-0.5 - mov, df, scale)
        p_home = 1 - _t_cdf(0.5 - mov, df, scale)
    else:
        sigma = (err or {}).get("sd") or 21.0
        p_away = phi((-0.5 - mov) / sigma)
        p_home = 1 - phi((0.5 - mov) / sigma)
    return p_home, max(0.0, 1 - p_home - p_away), p_away


def devig(a, b):
    if not a or not b:
        return None, None
    ia, ib = 1 / a, 1 / b
    return ia / (ia + ib), ib / (ia + ib)


def latest_lines(odds_rows):
    """Consensus line per fixture from its newest snapshot.

    The log holds one row per book, so a single row is one book's opinion.
    Within the newest snapshot use Pinnacle if it is there, else the median
    across books.
    """
    snaps = {}
    for r in odds_rows:
        if not r.get("home_spread"):
            continue
        key = (norm(r["home_team"]), norm(r["away_team"]))
        snaps.setdefault(key, {}).setdefault(r["snapshot"], []).append({
            "book": r["book"], "commence": r.get("commence", ""),
            "spread": float(r["home_spread"]),
            "total": float(r["total"]) if r.get("total") else None,
            "h2h_home": float(r["h2h_home"]) if r.get("h2h_home") else None,
            "h2h_away": float(r["h2h_away"]) if r.get("h2h_away") else None,
        })

    def consensus(books, snapshot):
        sharp = [b for b in books if "pinnacle" in b["book"]]
        if sharp:
            rec = dict(sharp[0])
            rec["book"] = "pinnacle"
        else:
            rec = dict(books[0])
            rec["book"] = f"median/{len(books)}" if len(books) > 1 else books[0]["book"]
            for fld in ("spread", "total", "h2h_home", "h2h_away"):
                vals = sorted(b[fld] for b in books if b[fld] is not None)
                rec[fld] = vals[len(vals) // 2] if vals else None
        rec["snapshot"] = snapshot
        return rec

    latest, first = {}, {}
    for key, by_snap in snaps.items():
        order = sorted(by_snap)
        latest[key] = consensus(by_snap[order[-1]], order[-1])
        first[key] = consensus(by_snap[order[0]], order[0])
    return latest, first


def played_map(results_rows):
    """(home, away) -> [(date, home_score, away_score), ...].

    Sides can meet more than once a season, so every meeting is kept and
    callers pick by date; matching on the pairing alone would settle this
    round's fixture against an earlier meeting.
    """
    out = {}
    for r in results_rows:
        key = (norm(r["home_team"]), norm(r["away_team"]))
        out.setdefault(key, []).append(
            (r["date"], int(r["home_score"]), int(r["away_score"])))
    return out


def result_since(played, key, floor_date):
    """Score for the first meeting on/after floor_date, else None."""
    for date, hs, as_ in sorted(played.get(key, [])):
        if floor_date is None or date >= floor_date:
            return (hs, as_)
    return None


boards, meta_leagues = {}, {}
played_all, latest_all = {}, {}

for lg, cfg in LEAGUES.items():
    params = json.load(open(os.path.join(BASE, cfg["params"]), encoding="utf-8"))
    sigma = params["oos_margin_rmse"]
    slope = params["margin_cal"][1]
    err = params.get("error_model")
    fairs = read_csv(cfg["fairs"])
    odds = read_csv(cfg["odds"])
    latest, first = latest_lines(odds)
    played = played_map(read_csv(cfg["results"]))
    for k, v in played.items():
        played_all.setdefault(k, []).extend(v)
    latest_all.update(latest)

    rows = []
    for f in fairs:
        home, away = norm(f["home_team"]), norm(f["away_team"])
        key = (home, away)
        raw = float(f["raw_margin"])
        mov = float(f["fair_mov"])
        ln = latest.get(key)
        spread = ln["spread"] if ln else (
            float(f["market_line"]) if f.get("market_line") else None)
        p_h, p_d, p_a = win_probs(mov, err)
        ip_h = ip_a = None
        if ln and ln["h2h_home"] and ln["h2h_away"]:
            ip_h, ip_a = devig(ln["h2h_home"], ln["h2h_away"])
        score = result_since(played, key, f.get("generated"))
        rows.append({
            "home": home, "away": away,
            "kickoff": ln["commence"] if ln else "",
            "raw": raw, "mov": mov,
            "fair_total": float(f["fair_total"]),
            "spread": spread, "book": ln["book"].replace("_manual", "") if ln else None,
            "total": ln["total"] if ln else None,
            "edge": (raw + spread) if spread is not None else None,
            "p_home": p_h, "p_draw": p_d, "p_away": p_a,
            "ip_home": ip_h, "ip_away": ip_a,
            "score": score,
        })
    rows.sort(key=lambda r: (r["score"] is not None, r["kickoff"] or "z"))
    boards[cfg["title"]] = rows
    meta_leagues[cfg["title"]] = {
        "round": cfg["round"], "slope": slope, "sigma": sigma,
        "oos": params["oos_games"], "lam_form": params["lam_form"],
        "err_dist": (err or {}).get("dist", "norm"),
        "err_df": (err or {}).get("df"),
    }

# ---- ledger -------------------------------------------------------------
# Risk/win convention: every bet is sized to win TARGET, so the stake is
# TARGET/(price-1) and varies with the price. A win returns TARGET; a loss
# costs the stake. At 1.87 (-115 US) that is $114.94 risked to win $100.
TARGET = 100.0
odds_by_league = {lg: read_csv(cfg["odds"]) for lg, cfg in LEAGUES.items()}


def american(dec):
    return round((dec - 1) * 100) if dec >= 2 else -round(100 / (dec - 1))


def entry_price(league, home, away, sel, line, logged):
    """Best price actually on offer for this selection/line at entry.

    Returns (price, source). Falls back to the ledger's own price, marked
    "assumed", when the log has no priced quote — SL lines were captured by
    hand without prices.
    """
    cands = []
    for r in odds_by_league[league]:
        if norm(r["home_team"]) != home or norm(r["away_team"]) != away:
            continue
        if not r.get("home_spread") or r["snapshot"][:10] < logged:
            continue
        sp = float(r["home_spread"])
        if sel == home and sp == line and r.get("home_spread_price"):
            cands.append((r["snapshot"], float(r["home_spread_price"]), r["book"]))
        elif sel == away and -sp == line and r.get("away_spread_price"):
            cands.append((r["snapshot"], float(r["away_spread_price"]), r["book"]))
    if not cands:
        return None, None
    earliest = min(c[0] for c in cands)
    best = max((c for c in cands if c[0] == earliest), key=lambda c: c[1])
    return best[1], best[2].replace("_manual", "")


settled, open_bets = [], []
for b in read_csv("rugby_bets.csv"):
    home, away = norm(b["home_team"]), norm(b["away_team"])
    sel, line = norm(b["selection"]), float(b["line"])
    market = (b.get("market") or "spread").strip().lower()
    is_total = market == "total"
    if is_total:
        price, price_src, assumed = float(b["price"]), b["book"], True
    else:
        price, price_src = entry_price(b["league"], home, away, sel, line, b["logged"])
        assumed = price is None
        if assumed:
            price, price_src = float(b["price"]), "assumed"
    key = (home, away)
    ln = latest_all.get(key)
    # CLV: how many points better than the current/closing number we took.
    # Under wants a higher line, Over and spreads want the number they took
    # to beat where the market ended up.
    clv = None
    if ln is not None:
        if is_total:
            if ln["total"] is not None:
                clv = round((line - ln["total"]) if sel.lower() == "under"
                            else (ln["total"] - line), 2)
        else:
            cur = ln["spread"] if sel == home else -ln["spread"]
            clv = round(line - cur, 2)
    score = result_since(played_all, key, b["logged"])
    stake = TARGET / (price - 1)
    rec = {"league": b["league"].upper(), "home": home, "away": away,
           "sel": sel, "line": line, "market": market, "price": price,
           "us": american(price), "stake": round(stake, 2), "book": price_src,
           "assumed": assumed, "clv": clv, "note": b["note"],
           "logged": b["logged"]}
    if score:
        if is_total:
            # margin-to-line for a total: how far the result beat the number
            ats = (line - (score[0] + score[1])) if sel.lower() == "under" \
                else ((score[0] + score[1]) - line)
        else:
            margin = (score[0] - score[1]) if sel == home else (score[1] - score[0])
            ats = margin + line
        rec["score"] = f"{score[0]}-{score[1]}"
        rec["ats"] = round(ats, 1)
        rec["result"] = "WIN" if ats > 0 else ("PUSH" if ats == 0 else "LOSS")
        rec["pnl"] = 0.0 if ats == 0 else (TARGET if ats > 0 else -stake)
        settled.append(rec)
    else:
        if ln is None:
            rec["cur"] = None
        elif is_total:
            rec["cur"] = ln["total"]
        else:
            rec["cur"] = ln["spread"] if sel == home else -ln["spread"]
        open_bets.append(rec)

pnl = sum(b["pnl"] for b in settled)
risked = sum(b["stake"] for b in settled if b["result"] != "PUSH")
clvs = [b["clv"] for b in (settled + open_bets) if b["clv"] is not None]

P = {
    "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "boards": boards,
    "leagues": meta_leagues,
    "settled": settled,
    "open": open_bets,
    "meta": {
        "record": f"{sum(1 for b in settled if b['result']=='WIN')}-"
                  f"{sum(1 for b in settled if b['result']=='LOSS')}",
        "pnl": round(pnl, 2),
        "roi": round(pnl / risked * 100, 1) if risked else 0.0,
        "avg_clv": round(sum(clvs) / len(clvs), 2) if clvs else None,
        "n_clv": len(clvs),
        "target": TARGET,
        "risked": round(risked, 2),
    },
}

HTML = """<title>Rugby League Fair Value Board</title>
<style>
:root{--paper:#F3F5F7;--ink:#1C2733;--sub:#5E6E7D;--rule:#D6DDE3;--brass:#9A711F;
--card:#FFFFFF;--card2:#EDF1F4;--steel:#58748C;--mist:#C2CCD5;--sand:#B08A57;
--pos:#2F7D5B;--neg:#B4453C;--chip:#E7ECF0;}
@media (prefers-color-scheme:dark){:root{--paper:#0F151C;--ink:#E6ECF1;--sub:#8FA0AF;
--rule:#26313D;--brass:#D9A93C;--card:#161E27;--card2:#1B2530;--steel:#5C82A2;
--mist:#3A4854;--sand:#B98D52;--pos:#5BB98C;--neg:#D9736A;--chip:#1D2833;}}
:root[data-theme="dark"]{--paper:#0F151C;--ink:#E6ECF1;--sub:#8FA0AF;--rule:#26313D;
--brass:#D9A93C;--card:#161E27;--card2:#1B2530;--steel:#5C82A2;--mist:#3A4854;
--sand:#B98D52;--pos:#5BB98C;--neg:#D9736A;--chip:#1D2833;}
:root[data-theme="light"]{--paper:#F3F5F7;--ink:#1C2733;--sub:#5E6E7D;--rule:#D6DDE3;
--brass:#9A711F;--card:#FFFFFF;--card2:#EDF1F4;--steel:#58748C;--mist:#C2CCD5;
--sand:#B08A57;--pos:#2F7D5B;--neg:#B4453C;--chip:#E7ECF0;}
*{box-sizing:border-box;}
body{background:var(--paper);color:var(--ink);margin:0;
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;padding:26px 20px 72px;}
.wrap{max-width:1100px;margin:0 auto;}
h1{font-size:23px;font-weight:700;letter-spacing:-0.01em;margin:0;}
.stamp{color:var(--sub);font-size:13px;margin-top:2px;}
.note{color:var(--sub);font-size:13px;margin:6px 0 22px;max-width:74ch;}
.note b{color:var(--ink);font-weight:600;}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
gap:12px;margin-bottom:24px;}
.c{background:var(--card);border:1px solid var(--rule);border-radius:9px;
padding:13px 15px;min-width:0;}
.c .k{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--sub);
font-weight:600;}
.c .v{font-size:25px;font-weight:700;margin-top:3px;font-variant-numeric:tabular-nums;}
.c .sub{font-size:12px;color:var(--sub);margin-top:1px;}
.c .v.pos{color:var(--pos);} .c .v.neg{color:var(--neg);}
.league{margin:30px 0 0;}
.eyebrow{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;
border-bottom:2px solid var(--brass);padding-bottom:6px;}
.eyebrow h2{font-size:13px;font-weight:700;text-transform:uppercase;
letter-spacing:.09em;margin:0;color:var(--brass);}
.eyebrow .meta{color:var(--sub);font-size:12.5px;}
.scroller{overflow-x:auto;}
table{border-collapse:collapse;width:100%;min-width:900px;background:var(--card);}
th{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--sub);
font-weight:600;text-align:right;padding:10px 10px 6px;border-bottom:1px solid var(--rule);
white-space:nowrap;}
th.l{text-align:left;}
td{padding:9px 10px;border-bottom:1px solid var(--rule);text-align:right;
white-space:nowrap;font-family:ui-monospace,"Cascadia Mono",Consolas,monospace;
font-variant-numeric:tabular-nums;font-size:13.5px;}
td.l{text-align:left;font-family:inherit;font-size:14px;min-width:0;}
tr:hover td{background:var(--card2);}
.date{color:var(--sub);font-size:12px;font-family:ui-monospace,Consolas,monospace;}
.tag{display:inline-block;background:var(--chip);color:var(--sub);border-radius:4px;
padding:1px 7px;font-size:11.5px;font-weight:600;}
.match b{font-weight:650;} .match .v{color:var(--sub);padding:0 5px;font-size:12px;}
.favcell{min-width:170px;}
.pnums{display:flex;justify-content:space-between;gap:8px;font-size:13px;}
.pnums .fav{font-weight:700;}
.pnums .hn{color:var(--steel);} .pnums .an{color:var(--sand);} .pnums .dn{color:var(--sub);}
.probbar{display:flex;height:6px;border-radius:3px;overflow:hidden;margin-top:5px;}
.probbar span{display:block;height:100%;}
.probbar .h{background:var(--steel);} .probbar .d{background:var(--mist);}
.probbar .a{background:var(--sand);}
.hi{color:var(--brass);font-weight:700;}
.res{display:inline-block;border-radius:4px;padding:1px 8px;font-size:11.5px;
font-weight:700;letter-spacing:.03em;}
.res.win{background:var(--pos);color:var(--paper);}
.res.loss{background:var(--sub);color:var(--paper);opacity:.75;}
.pnl-pos{color:var(--pos);font-weight:700;} .pnl-neg{color:var(--sub);}
.legend{display:flex;gap:16px;margin:14px 2px 0;color:var(--sub);font-size:12.5px;
flex-wrap:wrap;}
.legend .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;}
footer{margin-top:34px;color:var(--sub);font-size:12.5px;max-width:80ch;}
code{background:var(--chip);padding:1px 5px;border-radius:4px;
font-family:ui-monospace,Consolas,monospace;font-size:12.5px;}
</style>
<div class="wrap">
  <h1>Rugby League Fair Value Board</h1>
  <div class="stamp" id="stamp"></div>
  <p class="note"><b>Fair margins</b> come from a two-stage OLS per league:
  stage&nbsp;1 predicts both scores from home/away split averages plus a decayed
  net-form term; stage&nbsp;2 calibrates the raw margin against walk-forward
  out-of-sample results. <b>Raw margin</b> is the number to hold against a
  spread; <b>fair MOV</b> is the shrunk single-game prediction, and the
  shrinkage factor is the model's earned conviction (NRL ~31%, SL ~75%).
  <b>Edge</b> = raw margin &minus; market-implied margin. Win probabilities are
  the fair MOV read through the model's own out-of-sample error. Totals carry
  no out-of-sample signal in either league and are shown for reference only.
  Everything here is paper until the CLV ledger proves the edges real.</p>

  <div class="cards" id="cards"></div>
  <div id="boards"></div>
  <div id="positions"></div>
  <div id="ledger"></div>

  <div class="legend">
    <span><span class="sw" style="background:var(--steel)"></span>Home win</span>
    <span><span class="sw" style="background:var(--mist)"></span>Draw</span>
    <span><span class="sw" style="background:var(--sand)"></span>Away win</span>
    <span><span class="sw" style="background:var(--brass)"></span>Edge &ge; 6 pts
      = bet-sized disagreement (4&ndash;6 = lean)</span>
    <span>margins in home-team terms &middot; kickoffs local to each league</span>
  </div>
  <footer>Rebuild with <code>python rugby_board.py</code> after
  <code>python rl_results_pull.py</code>, <code>python nrl_odds_pull.py</code> and
  <code>python nrl_ols_deviation_model.py [nrl|sl]</code>. Lines: The Odds API
  (NRL, 12 AU books) plus manual Pinnacle/Bet365 captures; results scraped from
  rugbyleagueproject.org. Ledger is paper on a <b>risk/win $100</b> basis: each
  bet is sized to win $100 at the best price on offer when it was logged, so the
  stake is 100/(price&minus;1) &mdash; $114.94 at 1.87 (&minus;115). A win returns
  $100, a loss costs the stake, and ROI is P/L over the total risked.</footer>
</div>
<script>
const P = __DATA__;
const f1 = v => (v>=0?"+":"") + v.toFixed(1);
const pct = v => (v*100).toFixed(0);
const now = new Date();

const tzf = {};
function ko(iso, tz) {
  if (!iso) return "--";
  if (!tzf[tz]) tzf[tz] = new Intl.DateTimeFormat("en-GB", {timeZone: tz,
    weekday: "short", hour: "2-digit", minute: "2-digit", hourCycle: "h23"});
  return tzf[tz].format(new Date(iso)).replace(",", "");
}

const m = P.meta;
const openN = P.open.length;
const cards = [
  {k:"Open positions", v:openN,
   sub:`${P.open.filter(b=>b.league==="NRL").length} NRL · ${P.open.filter(b=>b.league==="SL").length} Super League`},
  {k:"Settled record", v:m.record,
   sub:`risk/win $${m.target} · $${m.risked.toFixed(0)} risked`},
  {k:"Ledger P/L", v:(m.pnl>=0?"+$":"-$")+Math.abs(m.pnl).toFixed(0),
   pos:m.pnl>=0, neg:m.pnl<0, sub:"paper only — no stakes placed"},
  {k:"ROI", v:(m.roi>=0?"+":"")+m.roi.toFixed(1)+"%", pos:m.roi>=0, neg:m.roi<0,
   sub:"settled bets only"},
];
if (m.avg_clv !== null) cards.push({k:"Avg CLV", v:f1(m.avg_clv)+" pts",
  pos:m.avg_clv>=0, neg:m.avg_clv<0, sub:`${m.n_clv} bets v current/closing line`});
document.getElementById("cards").innerHTML = cards.map(c=>
  `<div class="c"><div class="k">${c.k}</div>
   <div class="v${c.pos?' pos':(c.neg?' neg':'')}">${c.v}</div>
   <div class="sub">${c.sub}</div></div>`).join("");
document.getElementById("stamp").textContent =
  Object.entries(P.leagues).map(([k,v])=>`${k} ${v.round}`).join(" · ")
  + ` · generated ${P.generated.slice(0,16).replace("T"," ")}Z`;

function boardRow(r, tz) {
  const done = r.score !== null && r.score !== undefined;
  const e = r.edge;
  const cls = e === null ? "" : (Math.abs(e) >= 6 ? "hi" : "");
  return `
    <td class="l"><span class="date">${done ? "FT" : ko(r.kickoff, tz)}</span></td>
    <td class="l match"><b>${r.home}</b><span class="v">v</span>${r.away}
      ${done ? `<span class="tag">${r.score[0]}–${r.score[1]}</span>` : ""}</td>
    <td class="l favcell">
      <div class="pnums">
        <span class="hn ${r.p_home>=r.p_away?"fav":""}">${pct(r.p_home)}</span>
        <span class="dn">${pct(r.p_draw)}</span>
        <span class="an ${r.p_away>r.p_home?"fav":""}">${pct(r.p_away)}</span>
      </div>
      <div class="probbar">
        <span class="h" style="width:${r.p_home*100}%"></span>
        <span class="d" style="width:${r.p_draw*100}%"></span>
        <span class="a" style="width:${r.p_away*100}%"></span>
      </div>
    </td>
    <td>${r.ip_home!=null ? pct(r.ip_home)+"/"+pct(r.ip_away) : "—"}</td>
    <td>${f1(r.raw)}</td>
    <td>${f1(r.mov)}</td>
    <td>${r.spread!=null ? f1(r.spread) : "—"}</td>
    <td class="l"><span class="tag">${r.book || "—"}</span></td>
    <td>${r.total!=null ? r.total.toFixed(1) : "—"}</td>
    <td class="${cls}">${e!=null ? f1(e) : "—"}</td>`;
}

const bd = document.getElementById("boards");
for (const [lg, rows] of Object.entries(P.boards)) {
  const meta = P.leagues[lg];
  const tz = lg === "NRL" ? "Australia/Sydney" : "Europe/London";
  const live = rows.filter(r => !r.score).length;
  const sec = document.createElement("section");
  sec.className = "league";
  sec.innerHTML = `<div class="eyebrow"><h2>${lg} — ${meta.round}</h2>
    <span class="meta">${live} to play · ${rows.length - live} settled ·
    conviction ${pct(meta.slope)}% · &plusmn;${meta.sigma.toFixed(0)} pt error ·
    ${meta.oos} OOS games · ${meta.err_dist==="t"
      ? `fat-tailed errors (t, df ${meta.err_df.toFixed(1)})`
      : "normal errors"}</span></div>`;
  const tbl = document.createElement("table");
  tbl.innerHTML = `<thead><tr>
    <th class="l">KO</th><th class="l">Match</th>
    <th class="l favcell">Model&nbsp;H / D / A</th><th>Mkt&nbsp;H/A</th>
    <th>Raw</th><th>Fair&nbsp;MOV</th><th>Spread</th><th class="l">Book</th>
    <th>Total</th><th>Edge</th></tr></thead>`;
  const tb = document.createElement("tbody");
  for (const r of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = boardRow(r, tz);
    tb.appendChild(tr);
  }
  tbl.appendChild(tb);
  const sc = document.createElement("div"); sc.className = "scroller";
  sc.appendChild(tbl); sec.appendChild(sc);
  bd.appendChild(sec);
}

// open positions
if (P.open.length) {
  const sec = document.createElement("section");
  sec.className = "league";
  const pos = P.open.filter(b=>b.clv>0).length;
  sec.innerHTML = `<div class="eyebrow"><h2>Open Positions</h2>
    <span class="meta">${P.open.length} live · ${pos} with the line already
    moved our way</span></div>`;
  const tbl = document.createElement("table");
  tbl.innerHTML = `<thead><tr><th class="l">Lg</th><th class="l">Match</th>
    <th class="l">Bet</th><th>Taken</th><th>Now</th><th>CLV</th>
    <th class="l">Book</th><th class="l">Note</th></tr></thead>`;
  const tb = document.createElement("tbody");
  for (const b of P.open) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="l"><span class="tag">${b.league}</span></td>
      <td class="l">${b.home} v ${b.away}</td>
      <td class="l"><b>${b.market==="total" ? b.sel+" "+b.line.toFixed(1) : b.sel+" "+f1(b.line)}</b>${b.market==="total"?' <span class="tag">total</span>':""}</td>
      <td>${b.market==="total" ? b.line.toFixed(1) : f1(b.line)}</td>
      <td>${b.cur!=null ? (b.market==="total"?b.cur.toFixed(1):f1(b.cur)) : "—"}</td>
      <td class="${b.clv>0?"pnl-pos":(b.clv<0?"pnl-neg":"")}">${b.clv!=null?f1(b.clv):"—"}</td>
      <td class="l"><span class="tag">${b.book}</span></td>
      <td class="l" style="white-space:normal;color:var(--sub);font-size:13px">${b.note}</td>`;
    tb.appendChild(tr);
  }
  tbl.appendChild(tb);
  const sc = document.createElement("div"); sc.className = "scroller";
  sc.appendChild(tbl); sec.appendChild(sc);
  document.getElementById("positions").appendChild(sec);
}

// settled ledger
if (P.settled.length) {
  const sec = document.createElement("section");
  sec.className = "league";
  sec.innerHTML = `<div class="eyebrow"><h2>Settled — CLV Ledger</h2>
    <span class="meta">${P.settled.length} graded · P/L
    ${(P.meta.pnl>=0?"+$":"-$")+Math.abs(P.meta.pnl).toFixed(0)}</span></div>`;
  const tbl = document.createElement("table");
  tbl.innerHTML = `<thead><tr><th class="l">Lg</th><th class="l">Match</th>
    <th class="l">Bet</th><th>Price</th><th class="l">@</th><th>Score</th>
    <th>ATS</th><th>CLV</th><th class="l">Result</th><th>Stake</th><th>P/L</th>
    </tr></thead>`;
  const tb = document.createElement("tbody");
  for (const b of P.settled) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="l"><span class="tag">${b.league}</span></td>
      <td class="l">${b.home} v ${b.away}</td>
      <td class="l"><b>${b.market==="total" ? b.sel+" "+b.line.toFixed(1) : b.sel+" "+f1(b.line)}</b>${b.market==="total"?' <span class="tag">total</span>':""}</td>
      <td>${b.price.toFixed(2)} <span class="date">${b.us>0?"+":""}${b.us}</span></td>
      <td class="l"><span class="tag"${b.assumed?' title="no priced quote logged — assumed"':""}>${b.book}</span></td>
      <td>${b.score}</td>
      <td class="${b.ats>0?"pnl-pos":"pnl-neg"}">${f1(b.ats)}</td>
      <td class="${b.clv>0?"pnl-pos":(b.clv<0?"pnl-neg":"")}">${b.clv!=null?f1(b.clv):"—"}</td>
      <td class="l"><span class="res ${b.result==="WIN"?"win":"loss"}">${b.result}</span></td>
      <td>$${b.stake.toFixed(2)}</td>
      <td class="${b.pnl>=0?"pnl-pos":"pnl-neg"}">${(b.pnl>=0?"+":"-")+"$"+Math.abs(b.pnl).toFixed(2)}</td>`;
    tb.appendChild(tr);
  }
  const tf = document.createElement("tfoot");
  const nAss = P.settled.filter(b=>b.assumed).length;
  tf.innerHTML = `<tr><td class="l" colspan="9" style="color:var(--sub)">
    ${P.settled.length} bets sized to win $${P.meta.target.toFixed(0)} each at the
    best price on offer at entry${nAss?` · ${nAss} priced by assumption (not captured)`:""}</td>
    <td>$${P.meta.risked.toFixed(2)}</td>
    <td class="${P.meta.pnl>=0?"pnl-pos":"pnl-neg"}">${(P.meta.pnl>=0?"+":"-")+"$"+Math.abs(P.meta.pnl).toFixed(2)}</td></tr>`;
  tbl.appendChild(tb);
  tbl.appendChild(tf);
  const sc = document.createElement("div"); sc.className = "scroller";
  sc.appendChild(tbl); sec.appendChild(sc);
  document.getElementById("ledger").appendChild(sec);
}
</script>
"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML.replace("__DATA__", json.dumps(P, separators=(",", ":"))))

print(f"wrote {OUT}")
print(f"  boards: " + ", ".join(f"{k} {len(v)} games" for k, v in boards.items()))
print(f"  open {len(open_bets)} · settled {len(settled)} · "
      f"record {P['meta']['record']} · P/L {P['meta']['pnl']:+.0f} · "
      f"ROI {P['meta']['roi']:+.1f}% · avg CLV {P['meta']['avg_clv']}")
