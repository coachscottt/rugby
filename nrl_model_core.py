"""Shared core for the NRL game scores deviation model (blended features).

Features per match (all leakage-free, from games BEFORE the match):
    HF/HA   = home team's flat avg points for/against in prior home games
    AF/AA   = away team's flat avg points for/against in prior away games
    HFORM   = home team's decayed avg net margin per game, all prior games
    AFORM   = away team's decayed avg net margin per game, all prior games
The flat splits carry stable team quality; the decayed net-margin terms
carry recent form (weight lam_form**k for a game k appearances ago).
lam_form=None drops the form terms (pure flat model).
"""

import os
import csv

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = {
    "nrl": rf"{BASE_DIR}\nrl_2026_results.csv",
    "sl": rf"{BASE_DIR}\sl_2026_results.csv",
}
PARAMS_PATH = {lg: rf"{BASE_DIR}\{lg}_model_params.json" for lg in RESULTS}
OUTPUT_PATH = {lg: rf"{BASE_DIR}\{lg}_2026_model_output.csv" for lg in RESULTS}
MIN_SPLIT = 3    # prior games required in a split before its average is used
MIN_TRAIN = 30   # training matches required before a round is predicted

FEATURES_BASE = ["intercept", "home_for", "home_against", "away_for", "away_against"]
FEATURES_FORM = ["home_form", "away_form"]


def feature_names(lam_form):
    return FEATURES_BASE + (FEATURES_FORM if lam_form is not None else [])


def load_matches(league="nrl"):
    with open(RESULTS[league], newline="", encoding="utf-8") as f:
        matches = list(csv.DictReader(f))
    for m in matches:
        m["home_score"] = int(m["home_score"])
        m["away_score"] = int(m["away_score"])
        m["round_no"] = int(m["round"].split()[1])
    matches.sort(key=lambda m: (m["round_no"], m["date"]))
    return matches


def wmean(vals, lam):
    w = lam ** np.arange(len(vals) - 1, -1, -1)
    return float(np.dot(w, vals) / w.sum())


def _vector(hh, ah, lam_form):
    x = [1.0, np.mean(hh["hf"]), np.mean(hh["ha"]),
         np.mean(ah["af"]), np.mean(ah["aa"])]
    if lam_form is not None:
        x += [wmean(hh["net"], lam_form), wmean(ah["net"], lam_form)]
    return x


def build_features(matches, lam_form=None):
    """Annotate each match with m['x'] (or None); return final team histories."""
    hist = {}
    for m in matches:
        hh = hist.setdefault(m["home_team"],
                             {"hf": [], "ha": [], "af": [], "aa": [], "net": []})
        ah = hist.setdefault(m["away_team"],
                             {"hf": [], "ha": [], "af": [], "aa": [], "net": []})
        if len(hh["hf"]) >= MIN_SPLIT and len(ah["af"]) >= MIN_SPLIT:
            m["x"] = _vector(hh, ah, lam_form)
        else:
            m["x"] = None
        hs, as_ = m["home_score"], m["away_score"]
        hh["hf"].append(hs); hh["ha"].append(as_); hh["net"].append(hs - as_)
        ah["af"].append(as_); ah["aa"].append(hs); ah["net"].append(as_ - hs)
    return hist


def team_features(hist, home, away, lam_form=None):
    return np.array(_vector(hist[home], hist[away], lam_form))


def lstsq(X, y):
    return np.linalg.lstsq(np.asarray(X, float), np.asarray(y, float), rcond=None)[0]


def ols(X, y):
    X = np.asarray(X, float); y = np.asarray(y, float)
    beta = lstsq(X, y)
    pred = X @ beta
    resid = y - pred
    n, k = X.shape
    sigma2 = resid @ resid / (n - k)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - (resid @ resid) / np.sum((y - y.mean()) ** 2)
    rmse = np.sqrt(np.mean(resid ** 2))
    return beta, se, pred, resid, r2, rmse


def walk_forward(usable):
    """Refit each round on prior rounds; return out-of-sample prediction records."""
    out = []
    for rnd in sorted({m["round_no"] for m in usable}):
        train = [m for m in usable if m["round_no"] < rnd]
        if len(train) < MIN_TRAIN:
            continue
        Xt = [m["x"] for m in train]
        bh = lstsq(Xt, [m["home_score"] for m in train])
        ba = lstsq(Xt, [m["away_score"] for m in train])
        for m in usable:
            if m["round_no"] == rnd:
                x = np.array(m["x"])
                ph, pa = float(x @ bh), float(x @ ba)
                out.append({"m": m, "pred_h": ph, "pred_a": pa,
                            "pm": ph - pa, "am": m["home_score"] - m["away_score"],
                            "pt": ph + pa, "at": m["home_score"] + m["away_score"]})
    return out
