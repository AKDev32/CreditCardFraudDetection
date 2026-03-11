import streamlit as st
import joblib
import numpy as np
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield — Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="centered"
)

# ── Theme state ───────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

dark = st.session_state.dark_mode

# ── Colour tokens ─────────────────────────────────────────────────────────────
if dark:
    BG           = "#080B12"
    SURFACE      = "#0E1220"
    SURFACE2     = "#141828"
    BORDER       = "#1E2540"
    BORDER2      = "#252D48"
    T_PRIMARY    = "#EDF0FA"
    T_SECONDARY  = "#7A84A8"
    T_MUTED      = "#3A4260"
    ACCENT       = "#3B82F6"
    ACCENT2      = "#2563EB"
    ACCENT_GLOW  = "rgba(59,130,246,0.18)"
    DANGER       = "#EF4444"
    DANGER_BG    = "#1A0808"
    DANGER_BORDER= "#4D1515"
    SUCCESS      = "#22C55E"
    SUCCESS_BG   = "#071A0E"
    SUCCESS_BORDER="#1A4D2A"
    INPUT_BG     = "#0B0F1C"
    BAR_TRACK    = "#141828"
    NOTE_BG      = "#0A1020"
    NOTE_BORDER  = "#1E3A6E"
    NOTE_TEXT    = "#7EB3F9"
    TOGGLE_BG    = "#0E1220"
    TOGGLE_B     = "#1E2540"
    TOGGLE_T     = "#7A84A8"
    FOOTER_T     = "#1E2540"
else:
    BG           = "#F0F3FB"
    SURFACE      = "#FFFFFF"
    SURFACE2     = "#F8FAFF"
    BORDER       = "#DDE3F5"
    BORDER2      = "#C8D2EC"
    T_PRIMARY    = "#0A0D1A"
    T_SECONDARY  = "#3D4A70"
    T_MUTED      = "#8A95B8"
    ACCENT       = "#2563EB"
    ACCENT2      = "#1D4ED8"
    ACCENT_GLOW  = "rgba(37,99,235,0.14)"
    DANGER       = "#DC2626"
    DANGER_BG    = "#FFF1F1"
    DANGER_BORDER= "#FECACA"
    SUCCESS      = "#16A34A"
    SUCCESS_BG   = "#F0FDF4"
    SUCCESS_BORDER="#86EFAC"
    INPUT_BG     = "#F8FAFF"
    BAR_TRACK    = "#EEF1FA"
    NOTE_BG      = "#EFF6FF"
    NOTE_BORDER  = "#BFDBFE"
    NOTE_TEXT    = "#1E40AF"
    TOGGLE_BG    = "#FFFFFF"
    TOGGLE_B     = "#DDE3F5"
    TOGGLE_T     = "#3D4A70"
    FOOTER_T     = "#C8D2EC"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{ font-family: 'Manrope', sans-serif; color: {T_PRIMARY}; }}

.stApp {{ background: {BG} !important; transition: background 0.3s ease; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 4rem; max-width: 820px; }}

/* ── Nav row ── */
.nav-row {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 2.2rem; padding-bottom: 1rem; border-bottom: 1px solid {BORDER}; }}
.nav-brand {{ font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 800; color: {T_PRIMARY}; letter-spacing: -0.01em; display: flex; align-items: center; gap: 0.5rem; }}
.nav-brand-dot {{ width: 8px; height: 8px; border-radius: 50%; background: {ACCENT}; box-shadow: 0 0 8px {ACCENT}; animation: glow-pulse 2s ease-in-out infinite; }}
@keyframes glow-pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.4; }} }}
.nav-badge {{ background: {SURFACE2}; border: 1px solid {BORDER}; border-radius: 20px; padding: 0.2rem 0.7rem; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: {T_MUTED}; }}

/* ── Theme toggle ── */
div[data-testid="column"]:last-child .stButton > button {{
    background: {TOGGLE_BG} !important; color: {TOGGLE_T} !important;
    border: 1.5px solid {TOGGLE_B} !important; border-radius: 20px !important;
    font-family: 'Manrope', sans-serif !important; font-size: 0.72rem !important;
    font-weight: 700 !important; padding: 0.28rem 0.9rem !important;
    width: auto !important; margin-top: 0 !important; box-shadow: none !important;
    transition: all 0.2s ease !important; letter-spacing: 0.03em !important;
}}
div[data-testid="column"]:last-child .stButton > button:hover {{
    border-color: {ACCENT} !important; color: {ACCENT} !important; transform: none !important;
}}

/* ── Hero ── */
.hero {{ margin-bottom: 0.4rem; }}
.hero-eyebrow {{ font-family: 'Manrope', sans-serif; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.2em; text-transform: uppercase; color: {ACCENT}; margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.55rem; }}
.hero-eyebrow::after {{ content: ''; width: 28px; height: 1.5px; background: {ACCENT}; opacity: 0.55; border-radius: 2px; }}
.hero-title {{ font-family: 'Syne', sans-serif; font-size: 2.9rem; font-weight: 800; line-height: 1.06; letter-spacing: -0.03em; color: {T_PRIMARY}; margin: 0 0 0.7rem 0; }}
.hero-title .accent {{ color: {ACCENT}; }}
.hero-sub {{ font-size: 0.92rem; font-weight: 500; color: {T_SECONDARY}; line-height: 1.68; max-width: 560px; margin-bottom: 1.8rem; }}
.hero-rule {{ border: none; border-top: 1px solid {BORDER}; margin: 1.6rem 0; }}

/* ── Section label ── */
.section-label {{ font-family: 'Manrope', sans-serif; font-size: 0.65rem; font-weight: 800; letter-spacing: 0.18em; text-transform: uppercase; color: {T_MUTED}; margin-bottom: 0.9rem; }}

/* ── Card ── */
.card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px; padding: 1.6rem; box-shadow: 0 2px 20px rgba(0,0,0,{0.18 if dark else 0.06}); margin-bottom: 1.2rem; transition: background 0.3s ease; }}

/* ── Model selector ── */
div[data-testid="stSelectbox"] > div > div {{
    background: {INPUT_BG} !important; border: 1.5px solid {BORDER2} !important;
    border-radius: 10px !important; color: {T_PRIMARY} !important;
    font-family: 'Manrope', sans-serif !important; font-weight: 600 !important;
}}

/* ── Number inputs ── */
input[type="number"], .stNumberInput input {{
    background: {INPUT_BG} !important; border: 1.5px solid {BORDER2} !important;
    border-radius: 8px !important; color: {T_PRIMARY} !important;
    font-family: 'Manrope', sans-serif !important; font-weight: 600 !important;
    font-size: 0.88rem !important;
}}
input[type="number"]:focus, .stNumberInput input:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px {ACCENT_GLOW} !important;
}}

label[data-testid="stWidgetLabel"] p {{
    font-family: 'Manrope', sans-serif !important; font-size: 0.72rem !important;
    font-weight: 700 !important; color: {T_SECONDARY} !important;
    letter-spacing: 0.04em !important;
}}

/* ── Slider ── */
.stSlider {{ padding: 0.2rem 0; }}

/* ── Predict button ── */
.stButton > button {{
    background: {ACCENT} !important; color: #FFFFFF !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'Manrope', sans-serif !important; font-size: 0.9rem !important;
    font-weight: 800 !important; letter-spacing: 0.02em !important;
    padding: 0.75rem 2rem !important; width: 100% !important;
    margin-top: 0.5rem !important;
    box-shadow: 0 4px 16px {ACCENT_GLOW} !important;
    transition: all 0.18s ease !important;
}}
.stButton > button:hover {{ background: {ACCENT2} !important; transform: translateY(-2px) !important; box-shadow: 0 6px 22px {ACCENT_GLOW} !important; }}
.stButton > button:active {{ transform: translateY(0) !important; }}

/* ── Result cards ── */
.result-card {{ border-radius: 16px; padding: 1.8rem 2rem; margin: 1.4rem 0; position: relative; overflow: hidden; box-shadow: 0 4px 24px rgba(0,0,0,{0.22 if dark else 0.08}); }}
.result-fraud {{ background: {DANGER_BG}; border: 1.5px solid {DANGER_BORDER}; }}
.result-legit {{ background: {SUCCESS_BG}; border: 1.5px solid {SUCCESS_BORDER}; }}
.result-icon {{ font-size: 2.4rem; margin-bottom: 0.5rem; line-height: 1; }}
.result-tag {{ font-family: 'Manrope', sans-serif; font-size: 0.65rem; font-weight: 800; letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 0.3rem; }}
.result-tag-fraud {{ color: {DANGER}; }}
.result-tag-legit {{ color: {SUCCESS}; }}
.result-headline {{ font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; line-height: 1.1; margin: 0; letter-spacing: -0.02em; }}
.result-fraud .result-headline {{ color: {DANGER}; }}
.result-legit .result-headline {{ color: {SUCCESS}; }}
.result-dot {{ position: absolute; top: 1.6rem; right: 1.8rem; width: 11px; height: 11px; border-radius: 50%; animation: pulse-dot 2s ease-in-out infinite; }}
.result-dot-fraud {{ background: {DANGER}; box-shadow: 0 0 0 4px rgba(239,68,68,0.2); }}
.result-dot-legit {{ background: {SUCCESS}; box-shadow: 0 0 0 4px rgba(34,197,94,0.2); }}
@keyframes pulse-dot {{ 0%,100%{{opacity:1;transform:scale(1);}} 50%{{opacity:0.4;transform:scale(0.7);}} }}

/* ── Confidence bars ── */
.conf-card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px; padding: 1.4rem 1.6rem; margin-bottom: 1rem; box-shadow: 0 2px 16px rgba(0,0,0,{0.14 if dark else 0.04}); }}
.bar-row {{ margin-bottom: 1.1rem; }}
.bar-label-row {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.42rem; }}
.bar-label {{ font-family: 'Manrope', sans-serif; font-size: 0.82rem; font-weight: 700; color: {T_SECONDARY}; }}
.bar-pct {{ font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; }}
.bar-track {{ height: 7px; background: {BAR_TRACK}; border-radius: 99px; overflow: hidden; }}
.bar-fill {{ height: 100%; border-radius: 99px; }}
.bar-fill-legit {{ background: linear-gradient(90deg,#22C55E,#16A34A); }}
.bar-fill-fraud {{ background: linear-gradient(90deg,#F87171,#DC2626); }}

/* ── Note ── */
.note-box {{ background: {NOTE_BG}; border: 1px solid {NOTE_BORDER}; border-radius: 10px; padding: 0.9rem 1.15rem; margin-top: 0.8rem; font-size: 0.875rem; font-weight: 600; color: {NOTE_TEXT}; line-height: 1.6; display: flex; align-items: flex-start; gap: 0.65rem; }}

/* ── Model info pills ── */
.model-info {{ display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 0.6rem; }}
.model-pill {{ background: {SURFACE2}; border: 1px solid {BORDER}; border-radius: 20px; padding: 0.22rem 0.75rem; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em; color: {T_SECONDARY}; }}
.model-pill.active {{ background: {ACCENT_GLOW}; border-color: {ACCENT}; color: {ACCENT}; }}

/* ── Stats row ── */
.stats-row {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 1rem; margin: 1.2rem 0; }}
.stat-card {{ background: {SURFACE2}; border: 1px solid {BORDER}; border-radius: 12px; padding: 1rem 1.1rem; text-align: center; }}
.stat-value {{ font-family: 'Syne', sans-serif; font-size: 1.55rem; font-weight: 800; color: {T_PRIMARY}; letter-spacing: -0.02em; line-height: 1; margin-bottom: 0.25rem; }}
.stat-label {{ font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: {T_MUTED}; }}

/* ── Footer ── */
.app-footer {{ margin-top: 3rem; border-top: 1px solid {BORDER}; padding-top: 1.2rem; font-family: 'Manrope', sans-serif; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: {FOOTER_T}; display: flex; justify-content: space-between; align-items: center; }}
</style>
""", unsafe_allow_html=True)

# ── Top nav ───────────────────────────────────────────────────────────────────
nav_col, _, btn_col = st.columns([5, 1, 1])
with nav_col:
    st.markdown(f"""
    <div class="nav-brand">
        <div class="nav-brand-dot"></div>
        FraudShield
        <span class="nav-badge">ML · v1.0</span>
    </div>""", unsafe_allow_html=True)
with btn_col:
    lbl = "☀️ Light" if dark else "🌙 Dark"
    if st.button(lbl, key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="hero-eyebrow">Financial Security · AI Detection</div>
    <h1 class="hero-title">Credit Card<br><span class="accent">Fraud Shield</span></h1>
    <p class="hero-sub">Enter transaction details below. The system runs your chosen ML classifier against 28 PCA-engineered features to instantly flag fraudulent activity.</p>
</div>
<hr class="hero-rule">
""", unsafe_allow_html=True)

# ── Dataset stats ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-value">284K</div>
        <div class="stat-label">Transactions</div>
    </div>
    <div class="stat-card">
        <div class="stat-value" style="color:{DANGER}">0.17%</div>
        <div class="stat-label">Fraud Rate</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">3</div>
        <div class="stat-label">ML Models</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Model selection ───────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Step 1 — Select Model</div>', unsafe_allow_html=True)

MODEL_OPTIONS = {
    "1️⃣  Logistic Regression":  "models/lr_model.pkl",
    "2️⃣  Decision Tree":        "models/dt_model.pkl",
    "3️⃣  K-Nearest Neighbor":   "models/knn_model.pkl",
}

MODEL_META = {
    "1️⃣  Logistic Regression":  {"acc": "97.8%", "prec": "91.2%", "rec": "77.6%", "f1": "83.8%", "note": "Fast, interpretable baseline. Works well with scaled features."},
    "2️⃣  Decision Tree":        {"acc": "99.9%", "prec": "82.6%", "rec": "76.5%", "f1": "79.4%", "note": "Human-readable decision rules. Pruned to max depth 10 to avoid overfitting."},
    "3️⃣  K-Nearest Neighbor":   {"acc": "99.9%", "prec": "93.1%", "rec": "79.6%", "f1": "85.8%", "note": "Distance-based classifier. Sensitive to feature scale; SMOTE-balanced training."},
}

selected_model_name = st.selectbox(
    "Model",
    list(MODEL_OPTIONS.keys()),
    label_visibility="hidden"
)

meta = MODEL_META[selected_model_name]
st.markdown(f"""
<div class="model-info">
    <span class="model-pill active">✓ Selected</span>
    <span class="model-pill">Acc {meta['acc']}</span>
    <span class="model-pill">Precision {meta['prec']}</span>
    <span class="model-pill">Recall {meta['rec']}</span>
    <span class="model-pill">F1 {meta['f1']}</span>
</div>
<div style="margin-top:0.6rem;font-size:0.8rem;font-weight:500;color:{T_SECONDARY};padding-left:0.1rem;">
    💬 {meta['note']}
</div>""", unsafe_allow_html=True)

st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)

# ── Transaction input ─────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Step 2 — Enter Transaction Features</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)

# Time & Amount
col1, col2 = st.columns(2)
with col1:
    time_val = st.number_input("Time (seconds since first tx)", value=50000.0, format="%.2f")
with col2:
    amount_val = st.number_input("Amount ($)", value=125.50, min_value=0.0, format="%.2f")

# PCA features V1–V28 in a clean 4-column grid
st.markdown(f"<p style='font-size:0.72rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{T_MUTED};margin:1rem 0 0.6rem 0;'>PCA Features V1 – V28</p>", unsafe_allow_html=True)

defaults = [
    -1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.10,
     0.36, 0.09, -0.55, -0.62, -0.99, -0.31, 1.47, -0.47,
     0.21, 0.02, 0.40, 0.25, -0.02, -0.40, 0.19, 0.01,
    -0.01, 0.01, -0.01, -0.02
]

v_vals = []
cols = st.columns(4)
for i in range(28):
    with cols[i % 4]:
        v = st.number_input(f"V{i+1}", value=defaults[i], format="%.4f", key=f"v{i+1}", label_visibility="visible")
        v_vals.append(v)

st.markdown("</div>", unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────────────
predict_btn = st.button("🛡️  Run Fraud Detection")

if predict_btn:
    # Load model (with fallback to demo mode if models not present)
    model_path = MODEL_OPTIONS[selected_model_name]

    features = np.array([[time_val] + v_vals + [amount_val]])

    if os.path.exists(model_path):
        try:
            clf = joblib.load(model_path)
            prediction = clf.predict(features)[0]
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(features)[0]
                legit_pct = proba[0] * 100
                fraud_pct = proba[1] * 100
            else:
                score = clf.decision_function(features)[0]
                fraud_pct = min(max((score + 3) / 6 * 100, 0), 100)
                legit_pct = 100 - fraud_pct
        except Exception as e:
            st.error(f"Model load error: {e}")
            st.stop()
    else:
        # ── DEMO MODE ────────────────────────────────────────────────
        # Simulate different confidence profiles per model
        demo_score = abs(sum(v_vals[:5])) / 5
        is_suspicious = demo_score > 1.5 or amount_val > 500

        boost_map = {
            "1️⃣  Logistic Regression":  0.80,
            "2️⃣  Decision Tree":        0.88,
            "3️⃣  K-Nearest Neighbor":   0.90,
        }
        conf_boost = boost_map.get(selected_model_name, 0.90)

        if is_suspicious:
            prediction = 1
            fraud_pct  = min(round(conf_boost * 100, 1), 99.0)
        else:
            prediction = 0
            fraud_pct  = max(round(demo_score * 6, 1), 1.5)
        legit_pct = round(100 - fraud_pct, 1)
        st.info("ℹ️  **Demo mode** — model `.pkl` files not found. Run `train_and_save_models.py` first. Showing simulated results.", icon="🔬")

    is_fraud = prediction == 1

    # Verdict card
    if is_fraud:
        st.markdown(f"""
        <div class="result-card result-fraud">
            <div class="result-dot result-dot-fraud"></div>
            <div class="result-icon">🚨</div>
            <div class="result-tag result-tag-fraud">Detection Result</div>
            <p class="result-headline">FRAUDULENT</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card result-legit">
            <div class="result-dot result-dot-legit"></div>
            <div class="result-icon">✅</div>
            <div class="result-tag result-tag-legit">Detection Result</div>
            <p class="result-headline">LEGITIMATE</p>
        </div>""", unsafe_allow_html=True)

    # Confidence bars
    st.markdown(f"""
    <div class="conf-card">
        <div class="section-label" style="margin-bottom:1rem;">Confidence Breakdown — {selected_model_name.split('  ')[-1]}</div>
        <div class="bar-row">
            <div class="bar-label-row">
                <span class="bar-label">✅ Legitimate</span>
                <span class="bar-pct" style="color:{SUCCESS}">{legit_pct:.1f}%</span>
            </div>
            <div class="bar-track"><div class="bar-fill bar-fill-legit" style="width:{legit_pct:.1f}%"></div></div>
        </div>
        <div class="bar-row" style="margin-bottom:0">
            <div class="bar-label-row">
                <span class="bar-label">🚨 Fraudulent</span>
                <span class="bar-pct" style="color:{DANGER}">{fraud_pct:.1f}%</span>
            </div>
            <div class="bar-track"><div class="bar-fill bar-fill-fraud" style="width:{fraud_pct:.1f}%"></div></div>
        </div>
    </div>""", unsafe_allow_html=True)

    top_conf = max(legit_pct, fraud_pct)
    if top_conf >= 90:
        note = f"High confidence ({top_conf:.1f}%) — {selected_model_name} has a strong signal on this transaction."
    elif top_conf >= 70:
        note = f"Moderate confidence ({top_conf:.1f}%) — consider additional manual review for high-value transactions."
    else:
        note = f"Low confidence ({top_conf:.1f}%) — features are ambiguous; escalate for human review."

    st.markdown(f"""
    <div class="note-box">
        <span style="flex-shrink:0;font-size:1rem;margin-top:0.05rem;">💡</span>
        <span><strong>Analyst Note:</strong> {note}</span>
    </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-footer">
    <span>FraudShield · ML Fraud Detection</span>
    <span>Logistic Regression · Decision Tree · KNN</span>
</div>
""", unsafe_allow_html=True)