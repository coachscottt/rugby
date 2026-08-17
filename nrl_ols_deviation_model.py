"""NRL game scores deviation model (OLS), two-stage, leakage-free, blended.

Stage 1 — OLS score models on flat season-split averages (stable team
quality) plus decayed net-form terms (recent form); see nrl_model_core.
The form decay lam_form is tuned by walk-forward out-of-sample margin
RMSE against the no-form baseline (lam_form = None).

Stage 2 — margin-of-victory mapping fit on walk-forward out-of-sample
predictions: actual_margin = a + b * pred_margin. Totals mapped likewise.

Outputs: fitted parameters to nrl_model_params.json (consumed by
predict_nrl_fixture.py) and per-match OOS predictions to
nrl_2026_model_output.csv.
"""

import csv
import json
import sys

import numpy as np

from nrl_model_core import (load_matches, build_features, walk_forward, ols,
                            feature_names, OUTPUT_PATH, PARAMS_PATH)

LEAGUE = sys.argv[1] if len(sys.argv) > 1 else "nrl"
OUTPUT = OUTPUT_PATH[LEAGUE]
PARAMS = PARAMS_PATH[LEAGUE]

FORM_GRID = [None, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

matches = load_matches(LEAGUE)
print(f"{LEAGUE}: {len(matches)} matches loaded")

# --- tune form decay on walk-forward OOS margin error ---
print("\n=== Form decay tuning (walk-forward OOS; None = flat baseline) ===")
print(f"{'lam_form':>9}{'n_oos':>7}{'raw_rmse':>10}{'cal_rmse':>10}{'slope':>8}{'r2':>7}")
results = {}
for lam in FORM_GRID:
    build_features(matches, lam)
    usable = [m for m in matches if m["x"] is not None]
    wf = walk_forward(usable)
    pm = np.array([r["pm"] for r in wf]); am = np.array([r["am"] for r in wf])
    bm, sem, _, _, r2, cal_rmse = ols(np.column_stack([np.ones(len(pm)), pm]), am)
    raw_rmse = np.sqrt(np.mean((am - pm) ** 2))
    results[lam] = {"wf": wf, "cal_rmse": cal_rmse}
    lbl = "flat" if lam is None else f"{lam:.2f}"
    print(f"{lbl:>9}{len(wf):>7}{raw_rmse:>10.2f}{cal_rmse:>10.2f}{bm[1]:>8.3f}{r2:>7.3f}")

best_lam = min(results, key=lambda l: results[l]["cal_rmse"])
print(f"\nSelected lam_form = {best_lam} (lowest calibrated OOS margin RMSE)")

# --- final fit at chosen form decay ---
hist = build_features(matches, best_lam)
usable = [m for m in matches if m["x"] is not None]
wf = results[best_lam]["wf"]
pm = np.array([r["pm"] for r in wf]); am = np.array([r["am"] for r in wf])
pt = np.array([r["pt"] for r in wf]); at = np.array([r["at"] for r in wf])

bm, sem, _, _, r2_m, rmse_m = ols(np.column_stack([np.ones(len(pm)), pm]), am)
bt, set_, _, _, r2_t, rmse_t = ols(np.column_stack([np.ones(len(pt)), pt]), at)

print(f"\n=== Stage 2 at lam_form={best_lam} ({len(wf)} OOS games) ===")
print(f"Margin: MOV   = {bm[0]:+.3f} + {bm[1]:.3f} * pred_margin   "
      f"(slope se {sem[1]:.3f}, R^2 {r2_m:.3f}, RMSE {rmse_m:.2f})")
print(f"Totals: total = {bt[0]:+.3f} + {bt[1]:.3f} * pred_total    "
      f"(slope se {set_[1]:.3f}, R^2 {r2_t:.3f}, RMSE {rmse_t:.2f})")

names = feature_names(best_lam)
Xf = np.array([m["x"] for m in usable])
bh_f, seh, _, _, r2_h, rmse_h = ols(Xf, [m["home_score"] for m in usable])
ba_f, sea, _, _, r2_a, rmse_a = ols(Xf, [m["away_score"] for m in usable])
for label, b, s, r2, rmse in [("home", bh_f, seh, r2_h, rmse_h),
                              ("away", ba_f, sea, r2_a, rmse_a)]:
    print(f"\n=== Final {label} score coefficients ({len(usable)} matches) ===")
    print(f"{'term':<14}{'coef':>9}{'std err':>9}{'t':>8}")
    for name, bi, si in zip(names, b, s):
        print(f"{name:<14}{bi:>9.3f}{si:>9.3f}{bi / si:>8.2f}")
    print(f"R^2 = {r2:.3f}   RMSE = {rmse:.2f} pts")

with open(PARAMS, "w", encoding="utf-8") as f:
    json.dump({
        "lam_form": best_lam,
        "beta_home": list(bh_f), "beta_away": list(ba_f),
        "margin_cal": list(bm), "total_cal": list(bt),
        "oos_games": len(wf), "usable_matches": len(usable),
        "oos_margin_rmse": rmse_m, "oos_total_rmse": rmse_t,
    }, f, indent=2)
print(f"\nParameters written to {PARAMS}")

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["round", "date", "home_team", "away_team",
                "home_score", "away_score",
                "pred_home", "pred_away", "pred_margin", "mov_calibrated",
                "actual_margin", "dev_margin",
                "pred_total", "total_calibrated", "actual_total"])
    by_id = {id(r["m"]): r for r in wf}
    for m in matches:
        r = by_id.get(id(m))
        base = [m["round"], m["date"], m["home_team"], m["away_team"],
                m["home_score"], m["away_score"]]
        if r:
            cal_m = bm[0] + bm[1] * r["pm"]
            cal_t = bt[0] + bt[1] * r["pt"]
            w.writerow(base + [round(r["pred_h"], 2), round(r["pred_a"], 2),
                               round(r["pm"], 2), round(cal_m, 2),
                               r["am"], round(r["am"] - cal_m, 2),
                               round(r["pt"], 2), round(cal_t, 2), r["at"]])
        else:
            w.writerow(base + ["", "", "", "", m["home_score"] - m["away_score"],
                               "", "", "", m["home_score"] + m["away_score"]])
print(f"Per-match OOS predictions written to {OUTPUT}")
