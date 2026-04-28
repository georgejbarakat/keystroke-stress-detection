"""
Streamlit app: Real-time cognitive stress prediction from keystroke dynamics.

Two ways to predict:
  1. Live typing capture — paste the user types here and we extract features.
  2. Manual feature entry — for demoing the model with arbitrary inputs.

Run with:  streamlit run app.py
"""
import time
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ──────────────────────────── Page setup ────────────────────────────
st.set_page_config(
    page_title="Keystroke Stress Detector",
    page_icon="⌨️",
    layout="wide",
)

# ─────────────────────────── Load artifacts ─────────────────────────
@st.cache_resource
def load_artifacts():
    bundle  = joblib.load("best_model.joblib")
    results = json.load(open("model_results.json"))
    return bundle, results

bundle, results = load_artifacts()
model         = bundle["model"]
FEATURES      = bundle["features"]
MODEL_NAME    = bundle["model_name"]
RANDOM_AUC    = bundle.get("random_cv_auc", float("nan"))
LOUO_AUC      = bundle.get("louo_cv_auc",   float("nan"))


# ────────────────────────── Header ──────────────────────────────────
st.title("⌨️ Predicting Cognitive Stress from Keystroke Dynamics")
st.caption(
    "DSCI-441 Final Project · Graham Phillips & George Barakat · "
    f"Best model: **{MODEL_NAME}**"
)

c1, c2, c3 = st.columns(3)
c1.metric("Random 5-fold CV AUC",       f"{RANDOM_AUC:.3f}")
c2.metric("Leave-one-user-out AUC",     f"{LOUO_AUC:.3f}")
c3.metric("Training windows",           f"{results['n_samples']}")

st.info(
    "**Honest interpretation:** the model performs well when training and "
    "testing data come from the same users (random CV), but drops to "
    "near-chance when evaluated on a held-out user. With only 2 users in "
    "the labeled dataset, the model is mostly learning individual typing "
    "habits rather than a universal stress signature. This is a known "
    "limitation discussed in our final report.",
    icon="⚠️",
)


# ──────────────────────── Input mode ───────────────────────────────
mode = st.sidebar.radio(
    "Input mode",
    ["Live typing capture", "Manual feature entry", "Model details"],
)

# ──────────────────── Helpers ───────────────────
def make_input_row(values: dict) -> pd.DataFrame:
    """Pad a partial dict with NaNs for any features not provided."""
    row = {f: values.get(f, np.nan) for f in FEATURES}
    return pd.DataFrame([row], columns=FEATURES)


def predict_and_display(values: dict):
    X = make_input_row(values)
    prob = float(model.predict_proba(X)[0, 1])
    pred = int(prob >= 0.5)

    pcol1, pcol2 = st.columns([1, 2])
    with pcol1:
        st.metric("P(stressed)", f"{prob:.1%}")
        if pred == 1:
            st.error("Predicted: **Stressed**")
        else:
            st.success("Predicted: **Not Stressed**")
    with pcol2:
        st.markdown("**Features used for this prediction:**")
        st.dataframe(X.T.rename(columns={0: "value"}), use_container_width=True)
    return prob, pred


# ════════════════════ Mode 1: live capture ═════════════════════════
if mode == "Live typing capture":
    st.subheader("Type below — we'll measure your typing rhythm")
    st.write(
        "Type at least 5–10 sentences naturally, then click "
        "**Predict from my typing**. The page captures keypress and release "
        "timestamps in your browser to compute dwell time, inter-key interval, "
        "typing speed, and error rate."
    )

    capture_html = """
    <style>
      #typebox {width: 100%; min-height: 160px; padding: 12px;
                font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                font-size: 14px; border: 2px solid #4a90a4; border-radius: 8px;}
      #stats  {margin-top: 10px; color: #444; font-family: monospace;
               font-size: 13px;}
      button.act {margin-top: 8px; margin-right: 8px; padding: 6px 14px;
                  background:#4a90a4; color:white; border:none; border-radius:6px;
                  cursor:pointer;}
      button.act:hover {background:#3a7080;}
    </style>
    <textarea id="typebox" placeholder="Start typing here..."></textarea>
    <div>
      <button class="act" onclick="window.computeFeats()">Compute features</button>
      <button class="act" onclick="window.resetCapture()">Reset</button>
    </div>
    <pre id="stats">No data yet — start typing.</pre>
    <script>
      let presses = {}, events = [];
      const box = document.getElementById('typebox');
      box.addEventListener('keydown', e => {
        if (!presses[e.code]) presses[e.code] = performance.now();
      });
      box.addEventListener('keyup', e => {
        const t0 = presses[e.code];
        if (t0 != null) {
          events.push({key: e.key, code: e.code, press: t0, release: performance.now()});
          delete presses[e.code];
        }
      });
      window.resetCapture = () => {
        events = []; presses = {};
        document.getElementById('typebox').value = '';
        document.getElementById('stats').innerText = 'No data yet — start typing.';
      };
      window.computeFeats = () => {
        if (events.length < 5) {
          document.getElementById('stats').innerText =
            'Need at least 5 keystrokes — keep typing.';
          return;
        }
        // Dwell times
        const dwell = events.map(e => e.release - e.press)
                            .filter(x => x > 0 && x < 2000);
        // Inter-key intervals (between consecutive presses)
        const presses_sorted = events.map(e => e.press).sort((a,b)=>a-b);
        const iki = [];
        for (let i=1; i<presses_sorted.length; i++) {
          const d = presses_sorted[i] - presses_sorted[i-1];
          if (d > 0 && d < 5000) iki.push(d);
        }
        const sum = a => a.reduce((s,x)=>s+x, 0);
        const mean = a => a.length ? sum(a)/a.length : NaN;
        const std  = a => {
          if (a.length < 2) return NaN;
          const m = mean(a);
          return Math.sqrt(sum(a.map(x=>(x-m)**2))/(a.length-1));
        };
        const median = a => {
          if (!a.length) return NaN;
          const s = [...a].sort((x,y)=>x-y);
          const m = Math.floor(s.length/2);
          return s.length%2 ? s[m] : (s[m-1]+s[m])/2;
        };
        const dur = (presses_sorted.at(-1)-presses_sorted[0])/1000;
        const speed = dur > 0 ? events.length/dur : 0;
        const errKeys = events.filter(e =>
          ['Backspace','Delete'].includes(e.key)).length;
        const errRate = errKeys / events.length;

        const F = {
          dwell_mean: mean(dwell), dwell_std: std(dwell), dwell_median: median(dwell),
          iki_mean: mean(iki), iki_std: std(iki),
          iki_cv: mean(iki) ? std(iki)/mean(iki) : NaN,
          iki_median: median(iki),
          typing_speed: speed, error_rate: errRate, n_events: events.length,
        };
        const txt = JSON.stringify(F, null, 2);
        document.getElementById('stats').innerText = txt;
        // Mirror into hidden input so user can paste/copy if needed
        navigator.clipboard && navigator.clipboard.writeText(txt).catch(()=>{});
      };
    </script>
    """
    components.html(capture_html, height=360)

    st.markdown(
        "After clicking **Compute features**, copy the JSON shown above "
        "(it's also auto-copied to your clipboard) and paste it below:"
    )
    pasted = st.text_area("Paste feature JSON here", height=160,
                          placeholder='{"dwell_mean": 120, "iki_mean": 200, ...}')

    if st.button("Predict from my typing", type="primary"):
        try:
            vals = json.loads(pasted)
            predict_and_display(vals)
        except Exception as e:
            st.error(f"Could not parse JSON: {e}")


# ════════════════════ Mode 2: manual entry ═════════════════════════
elif mode == "Manual feature entry":
    st.subheader("Enter feature values manually")
    st.write("Use this for demo purposes or to explore how predictions vary.")

    cols = st.columns(2)
    vals = {}
    defaults = {
        "dwell_mean": 130.0, "dwell_std": 80.0, "dwell_median": 110.0,
        "iki_mean": 250.0, "iki_std": 200.0, "iki_cv": 0.8, "iki_median": 200.0,
        "typing_speed": 4.0, "error_rate": 0.03,
        "mouse_speed_mean": 1500.0, "mouse_speed_std": 2500.0,
        "mouse_click_rate": 0.15, "mouse_move_rate": 25.0,
    }
    for i, f in enumerate(FEATURES):
        with cols[i % 2]:
            vals[f] = st.number_input(f, value=float(defaults.get(f, 0.0)),
                                       format="%.4f")

    if st.button("Predict", type="primary"):
        predict_and_display(vals)


# ════════════════════ Mode 3: details ══════════════════════════════
else:
    st.subheader("Model & evaluation details")
    st.markdown(f"**Selected model:** `{MODEL_NAME}`")
    st.markdown(f"**Best hyperparameters:** `{bundle['best_params']}`")

    st.markdown("### Performance comparison")
    rows = []
    for name, r in results["random_cv"].items():
        louo = results["leave_one_user_out"][name]["auc_mean"]
        rows.append({
            "Model": name,
            "Random 5-fold CV AUC": f"{r['auc_mean']:.3f} ± {r['auc_std']:.3f}",
            "95% CI":    f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]",
            "Leave-one-user-out AUC": f"{louo:.3f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### Features")
    st.markdown(
        "**Keystroke (9):** dwell mean/std/median, IKI mean/std/CV/median, "
        "typing speed, error rate.  \n"
        "**Mouse (4):** movement speed mean/std, click rate, move-event rate."
    )

    st.markdown("### Datasets")
    st.markdown(
        "- **Kaggle stress dataset** — 2 users, 27,233 keystroke events, "
        "98 self-reported condition labels (used for supervised training).\n"
        "- **IKDD keystroke dynamics** — 374 user sessions of raw inter-key "
        "intervals (used for distributional analysis and feature validation)."
    )

    st.markdown("### Key takeaways")
    st.markdown(
        "1. Random Forest with mouse features beats the M1 logistic-regression "
        "baseline by ~0.10 AUC under random CV.\n"
        "2. Under user-held-out CV, all models drop to ~0.53 AUC — the "
        "behavioral signal in this dataset is dominated by per-user habits, "
        "not generalizable stress markers.\n"
        "3. The fix to dwell-time outliers in this milestone produced "
        "physiologically plausible values (~150 ms median, vs the M1 bug of "
        "~110,000 ms).\n"
        "4. Future work: collect data from many more users so that "
        "user-held-out evaluation is statistically meaningful, and explore "
        "person-normalized features (z-scoring against each user's baseline)."
    )
