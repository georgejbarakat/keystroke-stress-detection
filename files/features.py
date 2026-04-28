"""
Feature engineering for the keystroke stress/fatigue project.

Fixes vs Milestone 1:
  - Dwell-time outlier filter (M1 had values ~110,000 ms because long pauses
    between sessions inflated single 'release_time - press_time' diffs).
  - Mouse-movement and click-rate features added.
  - Sliding-window labeling capped at MAX_LABEL_LOOKBACK to prevent windows
    being labeled by stale stress reports from hours earlier.
"""
import os
import numpy as np
import pandas as pd

# ---- Filters ---------------------------------------------------------------
DWELL_MAX_MS = 2000   # > 2s of holding a key is almost certainly an artifact
IKI_MAX_MS   = 5000   # gap of >5s between keystrokes => not active typing
MAX_LABEL_LOOKBACK_MIN = 30   # max minutes between window and its label

# ---- Loaders ---------------------------------------------------------------
def load_user_data(base_path, user):
    """Load keystrokes / condition / mouse for one user."""
    user_path = os.path.join(base_path, user)
    ks  = pd.read_csv(os.path.join(user_path, "keystrokes.tsv"),    sep="\t")
    uc  = pd.read_csv(os.path.join(user_path, "usercondition.tsv"), sep="\t")
    md  = pd.read_csv(os.path.join(user_path, "mousedata.tsv"),     sep="\t")
    for df in (ks, uc, md):
        df["user"] = user
    return ks, uc, md


def load_all(base_path, users):
    ks_all, uc_all, md_all = [], [], []
    for u in users:
        ks, uc, md = load_user_data(base_path, u)
        ks_all.append(ks); uc_all.append(uc); md_all.append(md)
    keystrokes_df = pd.concat(ks_all, ignore_index=True)
    condition_df  = pd.concat(uc_all, ignore_index=True)
    mouse_df      = pd.concat(md_all, ignore_index=True)

    keystrokes_df["Press_Time"]  = pd.to_datetime(keystrokes_df["Press_Time"], format="mixed")
    keystrokes_df["Relase_Time"] = pd.to_datetime(keystrokes_df["Relase_Time"], format="mixed")
    condition_df["Time"]         = pd.to_datetime(condition_df["Time"], format="mixed")
    mouse_df["Time"]             = pd.to_datetime(mouse_df["Time"], format="mixed")
    return keystrokes_df, condition_df, mouse_df


# ---- Labels ----------------------------------------------------------------
def binarize_stress(val):
    if pd.isna(val): return np.nan
    return 1 if "Stressed" in str(val) else 0

def binarize_fatigue(val):
    if pd.isna(val): return np.nan
    return 1 if str(val) in ["Low", "Below_Avg"] else 0

def add_labels(condition_df):
    condition_df = condition_df.copy()
    condition_df["stress_label"]  = condition_df["Stress_Val"].apply(binarize_stress)
    condition_df["fatigue_label"] = condition_df["Fatigue_Val"].apply(binarize_fatigue)
    return condition_df


# ---- Per-window feature extraction -----------------------------------------
def extract_keystroke_features(ks_window):
    """Behavioral features from a window of keystroke events."""
    if len(ks_window) < 5:
        return None

    # Dwell time (ms) -- filter outliers (FIX for M1 bug)
    dwell = (ks_window["Relase_Time"] - ks_window["Press_Time"]).dt.total_seconds() * 1000
    dwell = dwell[(dwell > 0) & (dwell < DWELL_MAX_MS)]

    # Inter-key interval (ms)
    iki = ks_window["Press_Time"].diff().dt.total_seconds().dropna() * 1000
    iki = iki[(iki > 0) & (iki < IKI_MAX_MS)]

    # Typing speed (keys/sec)
    duration = (ks_window["Press_Time"].max() - ks_window["Press_Time"].min()).total_seconds()
    speed = len(ks_window) / duration if duration > 0 else 0

    # Error rate (backspace/delete)
    error_keys = ks_window["Key"].str.lower().isin(["backspace", "delete"])
    error_rate = error_keys.sum() / len(ks_window)

    return {
        "dwell_mean":   dwell.mean()  if len(dwell) > 0 else np.nan,
        "dwell_std":    dwell.std()   if len(dwell) > 0 else np.nan,
        "dwell_median": dwell.median() if len(dwell) > 0 else np.nan,
        "iki_mean":     iki.mean()    if len(iki)   > 0 else np.nan,
        "iki_std":      iki.std()     if len(iki)   > 0 else np.nan,
        "iki_cv":       (iki.std() / iki.mean()) if (len(iki) > 0 and iki.mean() > 0) else np.nan,
        "iki_median":   iki.median()  if len(iki)   > 0 else np.nan,
        "typing_speed": speed,
        "error_rate":   error_rate,
        "n_keys":       len(ks_window),
    }


def extract_mouse_features(mouse_window):
    """Mouse behavior features for the same time window."""
    if len(mouse_window) < 3:
        return {
            "mouse_n_events": 0,
            "mouse_speed_mean": np.nan,
            "mouse_speed_std":  np.nan,
            "mouse_click_rate": 0.0,
            "mouse_move_rate":  0.0,
        }

    mw = mouse_window.sort_values("Time").copy()
    duration = (mw["Time"].max() - mw["Time"].min()).total_seconds()
    duration = duration if duration > 0 else 1.0

    # Movement speed (pixels per second) using consecutive Move events
    moves = mw[mw["Event_Type"] == "Move"].copy()
    speeds = []
    if len(moves) >= 2:
        dx = moves["X"].diff().abs()
        dy = moves["Y"].diff().abs()
        dt = moves["Time"].diff().dt.total_seconds()
        valid = (dt > 0) & (dt < 1.0)   # only consider near-consecutive samples
        dist = np.sqrt(dx**2 + dy**2)
        speeds = (dist[valid] / dt[valid]).values
        speeds = speeds[np.isfinite(speeds)]

    clicks = mw["Event_Type"].astype(str).str.contains("Click|Press", case=False, na=False).sum()
    moves_count = (mw["Event_Type"] == "Move").sum()

    return {
        "mouse_n_events":   len(mw),
        "mouse_speed_mean": float(np.mean(speeds)) if len(speeds) > 0 else np.nan,
        "mouse_speed_std":  float(np.std(speeds))  if len(speeds) > 0 else np.nan,
        "mouse_click_rate": clicks / duration,
        "mouse_move_rate":  moves_count / duration,
    }


# ---- Sliding-window dataset builder ----------------------------------------
def build_feature_dataset(keystrokes_df, condition_df, mouse_df=None,
                          window_minutes=10, overlap=0.5):
    """
    Slide a window over keystrokes and extract keystroke + mouse features,
    then join with the nearest preceding condition label (capped lookback).
    """
    records = []
    window_td  = pd.Timedelta(minutes=window_minutes)
    step_td    = pd.Timedelta(minutes=window_minutes * (1 - overlap))
    max_lookback = pd.Timedelta(minutes=MAX_LABEL_LOOKBACK_MIN)

    for user in keystrokes_df["user"].unique():
        ks_user = keystrokes_df[keystrokes_df["user"] == user].sort_values("Press_Time")
        uc_user = condition_df[condition_df["user"] == user].sort_values("Time")
        if mouse_df is not None:
            md_user = mouse_df[mouse_df["user"] == user].sort_values("Time")
        else:
            md_user = None

        if ks_user.empty or uc_user.empty:
            continue

        t = ks_user["Press_Time"].min()
        end_time = ks_user["Press_Time"].max()

        while t + window_td <= end_time:
            ks_w = ks_user[(ks_user["Press_Time"] >= t) &
                           (ks_user["Press_Time"] <  t + window_td)]
            ks_feats = extract_keystroke_features(ks_w)
            if ks_feats is None:
                t += step_td
                continue

            # Find nearest preceding condition label, with lookback cap
            window_end = t + window_td
            prior = uc_user[(uc_user["Time"] <= window_end) &
                            (uc_user["Time"] >= window_end - max_lookback)]
            if prior.empty:
                t += step_td
                continue
            nearest = prior.iloc[-1]

            rec = {**ks_feats}
            if md_user is not None:
                mw = md_user[(md_user["Time"] >= t) &
                             (md_user["Time"] <  t + window_td)]
                rec.update(extract_mouse_features(mw))

            rec["stress_label"]  = nearest["stress_label"]
            rec["fatigue_label"] = nearest["fatigue_label"]
            rec["user"]          = user
            rec["window_start"]  = t
            records.append(rec)

            t += step_td

    return pd.DataFrame(records)


# Feature column groups - exposed so other modules use the same lists
KEYSTROKE_FEATURES = [
    "dwell_mean", "dwell_std", "dwell_median",
    "iki_mean", "iki_std", "iki_cv", "iki_median",
    "typing_speed", "error_rate",
]

MOUSE_FEATURES = [
    "mouse_speed_mean", "mouse_speed_std",
    "mouse_click_rate", "mouse_move_rate",
]

ALL_FEATURES = KEYSTROKE_FEATURES + MOUSE_FEATURES
