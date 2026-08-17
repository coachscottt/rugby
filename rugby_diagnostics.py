"""Model-integrity checks for the rugby league boards.

1. Residual tails. The board's win probabilities read the fair margin through
   a normal error curve. Rugby league margins blow out more often than a
   normal allows, so fit a Student-t to the walk-forward out-of-sample
   residuals and test whether it beats the normal. The winning fit's
   parameters are written to {lg}_model_params.json for the board to use.

2. Totals. The totals stage reports a near-zero calibration slope, i.e. no
   out-of-sample signal. Re-test that properly, then separately ask whether
   captured market totals show a systematic over-pricing (the "SL unders keep
   landing" pattern) or whether that is a small-sample artefact.

    python rugby_diagnostics.py
"""

import csv
import json
import os

import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
LEAGUES = {"nrl": "NRL", "sl": "Super League"}


def read(path):
    with open(os.path.join(BASE, path), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def oos_rows(lg):
    return [r for r in read(f"{lg}_2026_model_output.csv") if r["dev_margin"]]


# ---------------------------------------------------------------- 1. tails
print("=" * 68)
print("1. RESIDUAL TAILS — normal vs Student-t on walk-forward OOS residuals")
print("=" * 68)

tail_fits = {}
for lg, title in LEAGUES.items():
    rows = oos_rows(lg)
    resid = np.array([float(r["dev_margin"]) for r in rows])
    n = len(resid)

    # normal MLE
    mu_n, sd_n = resid.mean(), resid.std(ddof=0)
    ll_norm = stats.norm.logpdf(resid, mu_n, sd_n).sum()

    # student-t MLE (df, loc, scale)
    df, loc, scale = stats.t.fit(resid)
    ll_t = stats.t.logpdf(resid, df, loc, scale).sum()

    # likelihood-ratio test: t nests normal at df -> inf, 1 extra parameter
    lr = 2 * (ll_t - ll_norm)
    p_lr = stats.chi2.sf(lr, 1) if lr > 0 else 1.0
    aic_n, aic_t = 2 * 2 - 2 * ll_norm, 2 * 3 - 2 * ll_t

    # how often do residuals exceed 2sd / 2.5sd vs what each model predicts
    z = (resid - mu_n) / sd_n
    obs2, obs25 = (np.abs(z) > 2).mean(), (np.abs(z) > 2.5).mean()
    exp2, exp25 = 2 * stats.norm.sf(2), 2 * stats.norm.sf(2.5)
    t2 = 2 * stats.t.sf(2 * sd_n / scale, df)
    t25 = 2 * stats.t.sf(2.5 * sd_n / scale, df)

    print(f"\n{title}  (n={n})")
    print(f"  normal:    sd {sd_n:5.2f}                      logL {ll_norm:8.2f}  AIC {aic_n:7.2f}")
    print(f"  student-t: scale {scale:5.2f}  df {df:5.2f}      logL {ll_t:8.2f}  AIC {aic_t:7.2f}")
    print(f"  LR test vs normal: chi2 {lr:5.2f}, p = {p_lr:.3f}"
          f"  -> {'t fits better' if aic_t < aic_n else 'normal is adequate'}")
    print(f"  |resid| > 2.0 sd: observed {obs2:5.1%}   normal says {exp2:5.1%}   t says {t2:5.1%}")
    print(f"  |resid| > 2.5 sd: observed {obs25:5.1%}   normal says {exp25:5.1%}   t says {t25:5.1%}")
    print(f"  excess kurtosis {stats.kurtosis(resid):+.2f} "
          f"(0 = normal), skew {stats.skew(resid):+.2f}")

    # The board centres the error curve on the fair margin rather than on the
    # fitted location. With left-skewed residuals the robust t location sits
    # above the mean, which would quietly assert that home sides beat the
    # number more than half the time at pick'em -- if that were real it would
    # be a bet in its own right, not a display detail. Mean residual is ~0 by
    # construction, so centre there and take only the tails from the fit.
    use_t = aic_t < aic_n
    tail_fits[lg] = {"dist": "t" if use_t else "norm", "df": float(df),
                     "scale": float(scale), "sd": float(sd_n), "n": n,
                     "fitted_loc": float(loc), "mean_resid": float(mu_n),
                     "skew": float(stats.skew(resid)),
                     "aic_norm": float(aic_n), "aic_t": float(aic_t)}
    print(f"  mean residual {mu_n:+.2f} (calibration should hold this near 0); "
          f"fitted t location {loc:+.2f} — not used, board centres on the fair margin")

# write the chosen error model into each league's params for the board
for lg, fit in tail_fits.items():
    path = os.path.join(BASE, f"{lg}_model_params.json")
    p = json.load(open(path, encoding="utf-8"))
    p["error_model"] = fit
    json.dump(p, open(path, "w", encoding="utf-8"), indent=2)
print(f"\nerror_model written to {'/'.join(LEAGUES)}_model_params.json")

# effect on a sample fixture
print("\nEffect on win probabilities, centred on the fair margin (old -> new):")
print(f"{'MOV':>6}", "".join(f"{t:>24}" for t in LEAGUES.values()))
for mov in (0, 3, 7, 14, 21):
    cells = ""
    for lg in LEAGUES:
        f = tail_fits[lg]
        pn = 1 - stats.norm.cdf(0.5, mov, f["sd"])
        new = (1 - stats.t.cdf(0.5, f["df"], mov, f["scale"])
               if f["dist"] == "t" else pn)
        cells += f"{pn:9.1%} ->{new:8.1%}       "
    print(f"{mov:>6}", cells)

# ---------------------------------------------------------------- 2. totals
print("\n" + "=" * 68)
print("2. TOTALS — does the model's total predict, and are market totals biased?")
print("=" * 68)

for lg, title in LEAGUES.items():
    rows = oos_rows(lg)
    pred = np.array([float(r["pred_total"]) for r in rows])
    act = np.array([float(r["actual_total"]) for r in rows])
    res = stats.linregress(pred, act)
    print(f"\n{title}  (n={len(rows)})  model total vs actual")
    print(f"  actual = {res.intercept:+.2f} + {res.slope:.3f} x predicted"
          f"   (se {res.stderr:.3f}, p = {res.pvalue:.3f}, R^2 = {res.rvalue**2:.4f})")
    print(f"  slope differs from 0? {'YES' if res.pvalue < 0.05 else 'no'}"
          f"   | actual totals: mean {act.mean():.1f}, sd {act.std(ddof=1):.1f}"
          f"   | model spread: sd {pred.std(ddof=1):.2f}")

print("\n--- market totals vs results (captured lines only) ---")
for lg, title in LEAGUES.items():
    lines = {}
    for r in read(f"{lg}_odds_log.csv"):
        if r.get("total"):
            lines[(r["home_team"], r["away_team"])] = float(r["total"])
    results = {}
    for r in read(f"{lg}_2026_results.csv"):
        results[(r["home_team"], r["away_team"])] = int(r["home_score"]) + int(r["away_score"])
    alias = {"Manly Warringah Sea Eagles": "Manly Sea Eagles",
             "Cronulla Sutherland Sharks": "Cronulla Sharks"}
    pairs = []
    for (h, a), tot in lines.items():
        key = (alias.get(h, h), alias.get(a, a))
        if key in results:
            pairs.append((tot, results[key]))
    if not pairs:
        print(f"\n{title}: no captured totals with results yet")
        continue
    tot = np.array([p[0] for p in pairs]); act = np.array([p[1] for p in pairs])
    under = (act < tot).sum(); over = (act > tot).sum()
    diff = act - tot
    # binomial test on under rate, and a t-test that mean error differs from 0
    bt = stats.binomtest(int(under), under + over, 0.5)
    tt = stats.ttest_1samp(diff, 0)
    print(f"\n{title}  (n={len(pairs)} games with a captured line)")
    print(f"  unders {under} / overs {over}   under rate {under/(under+over):.1%}"
          f"   (binomial p = {bt.pvalue:.3f})")
    print(f"  actual minus line: mean {diff.mean():+.1f}, sd {diff.std(ddof=1):.1f}"
          f"   (t-test p = {tt.pvalue:.3f})")
    verdict = ("signal worth tracking" if bt.pvalue < 0.05 or tt.pvalue < 0.05
               else "consistent with chance at this sample size")
    print(f"  verdict: {verdict}")
