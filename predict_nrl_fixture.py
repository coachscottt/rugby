"""Predict NRL margin of victory and total for upcoming fixtures.

Usage: python predict_nrl_fixture.py HOME AWAY [HOME AWAY ...]
Team names match on substring, e.g. "Dragons" "Dolphins".

Uses parameters fitted by nrl_ols_deviation_model.py (nrl_model_params.json):
flat season splits + decayed net-form terms -> stage-1 scores -> stage-2
calibrated margin of victory and total.
"""

import json
import sys

import numpy as np

from nrl_model_core import load_matches, build_features, team_features, PARAMS_PATH

args = sys.argv[1:]
LEAGUE = "nrl"
if args and args[0] in PARAMS_PATH:
    LEAGUE = args.pop(0)

with open(PARAMS_PATH[LEAGUE], encoding="utf-8") as f:
    p = json.load(f)
lam_form = p["lam_form"]
beta_h, beta_a = np.array(p["beta_home"]), np.array(p["beta_away"])
(a_m, b_m), (a_t, b_t) = p["margin_cal"], p["total_cal"]

matches = load_matches(LEAGUE)
hist = build_features(matches, lam_form)
teams = sorted(hist)


def find(name):
    hits = [t for t in teams if name.lower() in t.lower()]
    if len(hits) != 1:
        sys.exit(f"Team '{name}' -> {hits or 'no match'}; teams: {', '.join(teams)}")
    return hits[0]


if len(args) < 2 or len(args) % 2:
    sys.exit("Usage: python predict_nrl_fixture.py [nrl|sl] HOME AWAY [HOME AWAY ...]")

print(f"Model ({LEAGUE}): lam_form={lam_form}, {p['usable_matches']} matches, "
      f"{p['oos_games']} OOS calibration points")
print(f"MOV   = {a_m:+.3f} + {b_m:.3f} * raw_margin  (OOS RMSE {p['oos_margin_rmse']:.1f})")
print(f"Total = {a_t:+.3f} + {b_t:.3f} * raw_total   (OOS RMSE {p['oos_total_rmse']:.1f})\n")

for h_name, a_name in zip(args[::2], args[1::2]):
    home, away = find(h_name), find(a_name)
    x = team_features(hist, home, away, lam_form)
    ph, pa = float(x @ beta_h), float(x @ beta_a)
    raw_m, raw_t = ph - pa, ph + pa
    mov = a_m + b_m * raw_m
    tot = a_t + b_t * raw_t
    fav = home if mov > 0 else away
    print(f"{home} (H) v {away} (A)")
    line = (f"  splits: HF {x[1]:.2f}  HA {x[2]:.2f}  AF {x[3]:.2f}  AA {x[4]:.2f}")
    if lam_form is not None:
        line += f"   form: H {x[5]:+.2f}  A {x[6]:+.2f}"
    print(line)
    print(f"  pred scores {ph:.1f} - {pa:.1f}   raw margin {raw_m:+.1f}   raw total {raw_t:.1f}")
    print(f"  calibrated MOV {mov:+.2f} ({fav} by {abs(mov):.1f})   calibrated total {tot:.1f}\n")
