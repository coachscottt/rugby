# Rugby on the Chromebook — runbook

Same Linux terminal as the soccer/WNBA setups. First-timer notes are in
soccer's RUNBOOK_CHROMEBOOK.md (paste = Ctrl+Shift+V, `ls` = letters L-S,
passwords/tokens type invisibly, Ctrl+C un-sticks anything).

## Why this one is the easiest
Rugby has **no database and no big files** — results, odds logs, model
params, per-round fairs and the bet ledger are all small CSV/JSON *inside
this repo*. So git is the whole sync story: pull before, push after.
Nothing to carry, nothing to train, no Polars (pure numpy/scipy/requests).

## One-time setup (~2 min) — paste as ONE block
```
cd ~ && git clone https://github.com/coachscottt/rugby.git && cd rugby && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && grep THE_ODDS_API_KEY ~/soccer/.env >> .env && python rugby_board.py
```
**Success looks like** the last lines printing
`wrote .../rugby_board.html` then `record 12-5 · P/L +653 · ROI +34.2%`
(numbers will grow over time) — the board rebuilt from the CSVs that came
down with the repo, so everything works.
- The odds key is copied straight from soccer's `.env` (same key, same name).
- Pushing uses the git credential you already cached for WNBA — no prompts.
  If it ever asks: username `coachscottt`, password = your GitHub token,
  **typed at the prompt, never pasted into a chat.**

## Every session
```
cd ~/rugby && source .venv/bin/activate && git pull
```

## Round routine
```
python nrl_odds_pull.py        # log current NRL lines (run a few times pre-round to keep line movement)
python rl_results_pull.py      # after the games: pull NRL + Super League results
python nrl_ols_deviation_model.py nrl && python nrl_ols_deviation_model.py sl
python rugby_diagnostics.py    # refresh error models after a refit (optional)
python rugby_board.py          # rebuild the board html
git add -A && git commit -m "round update" && git push
```
Then `claude` in this folder: "publish the rugby fair value board" — it
republishes the existing artifact at
https://claude.ai/code/artifact/7ae756c4-fb0e-4e13-8e22-91f8349f779c
(same URL every time; pass that URL so it updates rather than minting a new
one). The old Lavish board is retired — fair value board only.
Grade bets by pasting results/lines into the chat; they land in
`rugby_bets.csv` — identical to the PC flow.

## Sync rule
**`git pull` before you work, `git push` after — on whichever machine.**
That's it. **This repo is the only copy that counts.** The Windows PC also
has a loose set of these files in `OneDrive/WTA Model/` from before the repo
existed; those are stale and must not be edited or copied over the repo —
on the PC, work in `WTA Model/rugby/` and pull first.

(Soccer's "one machine owns the DB" rule does not apply here;
WNBA's "cloud collector is canonical" doesn't either — rugby has no
automation, the repo is simply the shared folder.)

## If something goes wrong
| You see | Do |
|---|---|
| `command not found: python` | `source .venv/bin/activate` |
| `THE_ODDS_API_KEY not found` | `.env` missing the key: `grep THE_ODDS_API_KEY ~/soccer/.env >> .env` |
| `git push` rejected (non-fast-forward) | `git pull --rebase --autostash && git push` (someone/somewhere pushed first) |
| `git push` asks for a password | enter your GitHub token at the prompt (see setup) |
| results pull returns nothing | RLP not updated yet — normal right after full-time; retry later |
| board shows stale record | you forgot `git pull` — the ledger CSV is in the repo |
