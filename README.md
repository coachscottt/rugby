# Rugby League fair-value models (NRL + Super League)

Two-stage OLS deviation models producing fair spreads/totals per round,
compared to Odds API / Betfair lines, paper-tracked in `rugby_bets.csv`
with CLV. See project memory / DECISIONS for method. Everything is small
CSV + JSON — **the data lives in this repo**, so git is the sync
mechanism between machines (no database to carry).

## Files
| | |
|---|---|
| `nrl_model_core.py` | features, walk-forward, OLS (shared by both leagues) |
| `nrl_ols_deviation_model.py` | fit / refit -> `*_model_params.json` |
| `predict_nrl_fixture.py` | fairs for named fixtures |
| `nrl_odds_pull.py` | Odds API NRL lines -> `nrl_odds_log.csv` (appends; keeps line movement) |
| `rl_results_pull.py [nrl|sl]` | RLP results scrape -> `*_2026_results.csv` |
| `rugby_board.py` | builds `rugby_board.html` (Lavish board) from the CSVs |
| `rugby_diagnostics.py` | unders/overs signal tests |
| `rugby_bets.csv` | the paper ledger (graded by hand from owner feedback) |
| `nrl_r*_fairs.csv`, `sl_r*_fairs.csv` | per-round fair sheets |

## Run (any machine)
```
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # then paste THE_ODDS_API_KEY
python nrl_odds_pull.py          # NRL lines
python rl_results_pull.py        # results (both leagues)
python rugby_board.py            # rebuild board html
```
Because the CSV/JSON data is committed, **always `git pull` before running
and `git push` after** — that's what keeps the PC and Chromebook copies
in sync (see RUNBOOK_CHROMEBOOK.md).
