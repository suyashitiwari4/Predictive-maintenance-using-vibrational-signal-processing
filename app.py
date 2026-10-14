"""
app.py
------
Streamlit dashboard for the Predictive Maintenance project.

Run with:
    streamlit run app.py

Sections:
  1. Signal & Spectrum   — waveform, FFT spectrum, spectrogram (time-frequency)
  2. Feature Extraction  — full time/frequency-domain feature breakdown
  3. Health Comparison   — current signal vs a healthy baseline, side by side
  4. CWRU: Fault Type    — multiclass fault prediction
  5. IMS: Degradation    — degradation-stage prediction
  6. Model Performance   — feature importance
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from features import (  # noqa: E402
    bandpass_filter,
    extract_features,
    frequency_domain_features,
    compute_spectrogram,
)

# -----------------------------------------------------------------
# Page config + theme
# -----------------------------------------------------------------
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide", page_icon="🛠️")

ACCENT = "#00D4B4"       # teal — healthy / primary accent
WARNING = "#FFB020"      # amber — suspect
DANGER = "#FF4D5E"       # red — fault / near-failure
BG_CARD = "#151B2B"
TEXT_MUTED = "#9AA5B1"

PLOTLY_TEMPLATE = "plotly_dark"

st.markdown(f"""
<style>
    .stApp {{
        background: radial-gradient(circle at top left, #10182b 0%, #0a0f1c 60%);
    }}
    .metric-card {{
        background: {BG_CARD};
        border: 1px solid #26314a;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 8px;
    }}
    .metric-label {{
        color: {TEXT_MUTED};
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .metric-value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: white;
    }}
    h1, h2, h3 {{
        font-family: 'Segoe UI', sans-serif;
    }}
    .status-pill {{
        display: inline-block;
        padding: 6px 16px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1rem;
    }}
</style>
""", unsafe_allow_html=True)

st.title("🛠️ Predictive Maintenance — Vibration Signal Analysis")
st.caption("Digital Signal Processing + Machine Learning for industrial bearing health monitoring")

# -----------------------------------------------------------------
# Sidebar — input
# -----------------------------------------------------------------
st.sidebar.header("⚙️ Input")
uploaded = st.sidebar.file_uploader(
    "Upload a vibration signal (.csv or .txt, single column)", type=["csv", "txt"]
)
fs = st.sidebar.number_input("Sampling frequency (Hz)", value=12000, step=1000)
window_len = st.sidebar.slider("Analysis window length (samples)", 1024, 8192, 4096, step=1024)

st.sidebar.header("🤖 Models")
cwru_model_path = st.sidebar.text_input("CWRU model path", "models/cwru_model.joblib")
ims_model_path = st.sidebar.text_input("IMS model path", "models/ims_model.joblib")


def metric_card(label, value, suffix=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}{suffix}</div>
    </div>
    """, unsafe_allow_html=True)


def make_healthy_reference(fs, n):
    """Synthetic smooth reference signal representing a 'healthy' bearing —
    used only to give the Health Comparison tab something to compare against
    when no ground-truth healthy sample is supplied."""
    t = np.linspace(0, n / fs, n)
    rng = np.random.default_rng(42)
    return 0.4 * np.sin(2 * np.pi * 30 * t) + 0.05 * rng.standard_normal(n)


# -----------------------------------------------------------------
# Load / generate signal
# -----------------------------------------------------------------
if uploaded is not None:
    raw = pd.read_csv(uploaded, header=None)
    signal_full = raw.iloc[:, 0].values.astype(float)
    signal = bandpass_filter(signal_full, fs=fs)
    source_label = uploaded.name
else:
    st.sidebar.info("No file uploaded — showing a synthetic demo signal with an injected fault.")
    t = np.linspace(0, 1, int(fs))
    signal = (
        0.5 * np.sin(2 * np.pi * 30 * t)
        + 1.1 * np.sin(2 * np.pi * 157 * t)
        + 0.35 * np.sin(2 * np.pi * 314 * t)
        + 0.15 * np.random.randn(len(t))
    )
    source_label = "synthetic demo signal"

signal = signal[:window_len] if len(signal) >= window_len else signal
healthy_ref = make_healthy_reference(fs, len(signal))

st.markdown(f"**Analyzing:** `{source_label}`  •  **{len(signal)} samples**  •  **{fs} Hz**")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📈 Signal & Spectrum", "🧮 Feature Extraction", "⚖️ Health Comparison",
     "🔩 CWRU: Fault Type", "📉 IMS: Degradation", "🌟 Model Performance"]
)

# ===================================================================
# TAB 1 — Signal & Spectrum
# ===================================================================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Time-Domain Waveform")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=signal[:2000], mode="lines",
                                  line=dict(color=ACCENT, width=1.3), name="Signal"))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=320,
                           margin=dict(l=10, r=10, t=10, b=10),
                           xaxis_title="Sample", yaxis_title="Amplitude")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Frequency Spectrum (FFT)")
        feats, freqs, mag = frequency_domain_features(signal, fs)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=freqs, y=mag, mode="lines",
                                   fill="tozeroy", line=dict(color=WARNING, width=1.3)))
        fig2.add_vline(x=feats["dominant_freq"], line_dash="dash", line_color=DANGER,
                        annotation_text=f"dominant: {feats['dominant_freq']:.1f} Hz")
        fig2.update_layout(template=PLOTLY_TEMPLATE, height=320,
                            margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title="Frequency (Hz)", yaxis_title="Magnitude")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Spectrogram — Time-Frequency View")
    st.caption("Shows how the frequency content changes over time (a plain FFT only gives the average spectrum).")
    f_sp, t_sp, Sxx = compute_spectrogram(signal, fs)
    fig3 = go.Figure(data=go.Heatmap(
        z=10 * np.log10(Sxx + 1e-12), x=t_sp, y=f_sp,
        colorscale="Turbo", colorbar=dict(title="dB")
    ))
    fig3.update_layout(template=PLOTLY_TEMPLATE, height=350,
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="Time (s)", yaxis_title="Frequency (Hz)")
    st.plotly_chart(fig3, use_container_width=True)

# ===================================================================
# TAB 2 — Feature Extraction
# ===================================================================
with tab2:
    feature_dict = extract_features(signal, fs)
    time_keys = ["mean", "variance", "std", "rms", "energy", "peak", "peak_to_peak",
                 "crest_factor", "kurtosis", "skewness", "shape_factor",
                 "impulse_factor", "margin_factor", "clearance_factor"]
    freq_keys = ["dominant_freq", "spectral_energy", "spectral_centroid", "spectral_spread",
                 "spectral_std", "spectral_entropy", "spectral_kurtosis",
                 "band_power_low", "band_power_mid", "band_power_high"]

    st.subheader("Key Indicators")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("RMS", f"{feature_dict['rms']:.3f}")
    with c2: metric_card("Crest Factor", f"{feature_dict['crest_factor']:.2f}")
    with c3: metric_card("Kurtosis", f"{feature_dict['kurtosis']:.2f}")
    with c4: metric_card("Dominant Freq", f"{feature_dict['dominant_freq']:.1f}", " Hz")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Time-Domain Features")
        tvals = {k: feature_dict[k] for k in time_keys}
        fig = go.Figure(go.Bar(
            x=list(tvals.values()), y=list(tvals.keys()), orientation="h",
            marker_color=ACCENT
        ))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=440,
                           margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Frequency-Domain Features")
        fvals = {k: feature_dict[k] for k in freq_keys}
        fig2 = go.Figure(go.Bar(
            x=list(fvals.values()), y=list(fvals.keys()), orientation="h",
            marker_color=WARNING
        ))
        fig2.update_layout(template=PLOTLY_TEMPLATE, height=440,
                            margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Full feature table"):
        st.dataframe(pd.DataFrame([feature_dict]).T.rename(columns={0: "value"}))

# ===================================================================
# TAB 3 — Health Comparison (current vs healthy baseline)
# ===================================================================
with tab3:
    st.subheader("Current Signal vs. Healthy Baseline")
    st.caption(
        "Compares your uploaded/demo signal against a clean reference signal across "
        "the parameters most sensitive to bearing faults. Large deviations suggest "
        "abnormal vibration behavior."
    )

    current_feats = extract_features(signal, fs)
    healthy_feats = extract_features(healthy_ref, fs)

    compare_keys = ["rms", "crest_factor", "kurtosis", "impulse_factor",
                     "spectral_energy", "spectral_entropy", "dominant_freq"]

    # Normalize each parameter pair to [0, 1] so scales are comparable on one radar chart
    radar_current, radar_healthy = [], []
    for k in compare_keys:
        c_val, h_val = current_feats[k], healthy_feats[k]
        m = max(abs(c_val), abs(h_val), 1e-9)
        radar_current.append(abs(c_val) / m)
        radar_healthy.append(abs(h_val) / m)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=radar_healthy + radar_healthy[:1],
                                   theta=compare_keys + compare_keys[:1],
                                   fill="toself", name="Healthy Baseline",
                                   line_color=ACCENT, opacity=0.6))
    fig.add_trace(go.Scatterpolar(r=radar_current + radar_current[:1],
                                   theta=compare_keys + compare_keys[:1],
                                   fill="toself", name="Current Signal",
                                   line_color=DANGER, opacity=0.6))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=450,
                       polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                       margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Side-by-Side Parameter Comparison")
    comp_df = pd.DataFrame({
        "Parameter": compare_keys,
        "Healthy Baseline": [healthy_feats[k] for k in compare_keys],
        "Current Signal": [current_feats[k] for k in compare_keys],
    })
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=comp_df["Parameter"], y=comp_df["Healthy Baseline"],
                           name="Healthy Baseline", marker_color=ACCENT))
    fig2.add_trace(go.Bar(x=comp_df["Parameter"], y=comp_df["Current Signal"],
                           name="Current Signal", marker_color=DANGER))
    fig2.update_layout(template=PLOTLY_TEMPLATE, barmode="group", height=380,
                        margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Raw comparison table"):
        st.dataframe(comp_df.set_index("Parameter"))

# ===================================================================
# TAB 4 — CWRU Fault Type
# ===================================================================
with tab4:
    st.subheader("CWRU Fault-Type Prediction")
    if os.path.exists(cwru_model_path):
        bundle = joblib.load(cwru_model_path)
        clf, le, feat_names = bundle["model"], bundle["label_encoder"], bundle["feature_names"]
        feats_row = pd.DataFrame([extract_features(signal, fs)])[feat_names]
        pred = clf.predict(feats_row)[0]
        proba = clf.predict_proba(feats_row)[0]
        label = le.inverse_transform([pred])[0]

        color = ACCENT if label == "Healthy" else DANGER
        st.markdown(
            f'<span class="status-pill" style="background:{color}22;color:{color};'
            f'border:1px solid {color};">Predicted: {label}</span>',
            unsafe_allow_html=True,
        )
        st.write("")
        fig = go.Figure(go.Bar(x=le.classes_, y=proba, marker_color=[
            ACCENT if c == label else "#3a4664" for c in le.classes_
        ]))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=320,
                           yaxis_title="Confidence", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(
            f"No model found at '{cwru_model_path}'. "
            "Train one first with: python src/train_cwru.py --data_dir data/cwru"
        )

# ===================================================================
# TAB 5 — IMS Degradation Stage
# ===================================================================
with tab5:
    st.subheader("IMS Degradation-Stage Prediction")
    if os.path.exists(ims_model_path):
        bundle = joblib.load(ims_model_path)
        clf, le, feat_names = bundle["model"], bundle["label_encoder"], bundle["feature_names"]
        feats_row = pd.DataFrame([extract_features(signal, fs)])[feat_names]
        pred = clf.predict(feats_row)[0]
        proba = clf.predict_proba(feats_row)[0]
        label = le.inverse_transform([pred])[0]

        stage_colors = {"Normal": ACCENT, "Suspect": WARNING,
                         "Degraded": WARNING, "Near-Failure": DANGER}
        color = stage_colors.get(label, WARNING)
        st.markdown(
            f'<span class="status-pill" style="background:{color}22;color:{color};'
            f'border:1px solid {color};">Predicted stage: {label}</span>',
            unsafe_allow_html=True,
        )
        st.write("")
        fig = go.Figure(go.Bar(x=le.classes_, y=proba, marker_color=[
            stage_colors.get(c, WARNING) for c in le.classes_
        ]))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=320,
                           yaxis_title="Confidence", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(
            f"No model found at '{ims_model_path}'. "
            "Train one first with: python src/train_ims.py --data_dir data/ims/1st_test"
        )

    st.subheader("Health Trend Curve (full run)")
    st.caption(
        "For a full bearing run (not a single window), plot RMS or kurtosis across "
        "all IMS files in the timeline here — a rising trend indicates degrading health. "
        "Requires loading the full run via data_loading.load_ims_folder()."
    )

# ===================================================================
# TAB 6 — Model Performance
# ===================================================================
with tab6:
    st.subheader("Feature Importance")
    for path, name, color in [(cwru_model_path, "CWRU", ACCENT), (ims_model_path, "IMS", WARNING)]:
        if os.path.exists(path):
            bundle = joblib.load(path)
            clf, feat_names = bundle["model"], bundle["feature_names"]
            if hasattr(clf, "feature_importances_"):
                st.markdown(f"**{name} model**")
                imp = pd.Series(clf.feature_importances_, index=feat_names).sort_values(ascending=True)
                fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=color))
                fig.update_layout(template=PLOTLY_TEMPLATE, height=420,
                                   margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption(f"{name} model not found at '{path}' — train it to see feature importance.")
