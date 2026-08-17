# Rugby on the Chromebook — runbook

Same Linux terminal as the soccer setup. First-timer notes are in
soccer's RUNBOOK_CHROMEBOOK.md (paste = Ctrl+Shift+V, `ls` = letters, etc.)

## One-time setup
```
cd ~ && git clone https://github.com/coachscottt/rugby.git && cd rugby
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env && nano .env      # paste THE_ODDS_API_KEY=... , Ctrl+O Enter Ctrl+X
python rugby_board.py                  # smoke test: should print "wrote ... rugby_board.html"
```
(The odds key is the same one in soccer's `.env` — copy the line from there:
`grep THE_ODDS_API_KEY ~/soccer/.env >> .env` does it in one go.)

## Every session
```
cd ~/rugby && source .venv/bin/activate && git pull
```

## Round routine
```
python nrl_odds_pull.py        # log current NRL lines (run a few times pre-round for movement)
python rl_results_pull.py      # after games: pull results
python rugby_board.py          # rebuild the board
git add -A && git commit -m "round update" && git push
```
Then in Claude Code (`claude` in this folder): "post the rugby board on
Lavish" / grade bets from your feedback into `rugby_bets.csv` as usual.

## THE SYNC RULE (different from soccer!)
Rugby has NO database — all data is CSVs inside the repo. So the rule is
simply: **`git pull` before you work, `git push` after.** Do that on
whichever machine you use and both stay identical. (Soccer's
"one machine owns the DB" rule does not apply here.)
