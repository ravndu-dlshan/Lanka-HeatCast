"""
Lanka HeatCast — Ensemble Learning Dashboard
Next-day maximum temperature forecasting for Colombo, Kandy & Nuwara Eliya (Sri Lanka)
Bagging · Boosting · Stacking · Hybrid (ARIMA+RF, RF+ANN, Linear+RF)

"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ART = Path(__file__).parent / "artifacts"
st.set_page_config(page_title="Lanka HeatCast · Ensemble Dashboard", page_icon="🌦️", layout="wide")

CITY_COLORS = {"Colombo": "#1f77b4", "Kandy": "#2ca02c", "Nuwara Eliya": "#9467bd"}
FAM_COLORS = {"Baseline": "#9e9e9e", "Single model": "#bdbdbd", "Bagging": "#2ca02c", "Boosting": "#ff7f0e",
              "Neural network": "#8c564b", "Stacking": "#1f77b4", "Time-series": "#17becf", "Hybrid": "#d62728"}
CITY_INFO = {"Colombo": (6.9271, 79.8612, "Coastal lowlands · 9 m"),
             "Kandy": (7.2906, 80.6337, "Mid-country hills · ~500 m"),
             "Nuwara Eliya": (6.9497, 80.7891, "Central highlands · ~1,900 m")}

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.hero {background: linear-gradient(110deg,#0b3d5c 0%,#127a6b 55%,#e8a33d 100%); color:white;
       padding:22px 28px; border-radius:16px; margin-bottom:14px;}
.hero h1 {margin:0; font-size:2.0rem;} .hero p {margin:4px 0 0 0; opacity:.92; font-size:1.02rem;}
.card {border:1px solid rgba(128,128,128,.25); border-radius:12px; padding:14px 16px; height:100%;}
.card h4 {margin:0 0 6px 0;}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load():
    lb = pd.read_csv(ART / "leaderboard.csv", index_col=0)
    pc = pd.read_csv(ART / "per_city_mae.csv", index_col=0)
    ss = pd.read_csv(ART / "season_mae.csv", index_col=0)
    pr = pd.read_csv(ART / "test_predictions.csv", parse_dates=["date", "forecast_for"])
    fi = pd.read_csv(ART / "feature_importance.csv", index_col=0)
    bc = pd.read_csv(ART / "bagging_curve.csv")
    tf = pd.read_csv(ART / "test_features.csv.gz", parse_dates=["date"])
    wd = pd.read_csv(ART / "weather_daily.csv.gz", parse_dates=["date"])
    meta = json.load(open(ART / "meta.json"))
    return lb, pc, ss, pr, fi, bc, tf, wd, meta


@st.cache_resource
def load_models():
    return joblib.load(ART / "models.joblib")


lb, per_city, season, preds, fimp, bcurve, tfeat, wdaily, meta = load()
MODELS = [m for m in lb.index]
best = lb.index[0]
pers = lb.loc["Persistence (baseline)", "MAE"]

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## 🌦️ Lanka HeatCast")
    st.caption("Ensemble learning for tomorrow's max temperature in Sri Lanka")
    city = st.selectbox("City", list(CITY_INFO), index=0)
    default_models = [best, "Random Forest", "CatBoost", "Persistence (baseline)"]
    chosen = st.multiselect("Models to compare", MODELS, default=[m for m in default_models if m in MODELS])
    st.divider()
    st.markdown("**Data:** Open-Meteo historical archive (ERA5), 2016-01-01 → 2026-11-15, 3 cities × 3,972 days")
    st.markdown(f"**Train:** 2016–2023 · **Test:** {meta['split']} → 2026-11-14 ({meta['n_test']:,} forecasts)")
    

st.markdown(f"""
<div class="hero"><h1>🌦️ Lanka HeatCast</h1>
<p>How hot will it be tomorrow in <b>Colombo</b>, <b>Kandy</b> and <b>Nuwara Eliya</b>? Ten years of Sri Lankan weather,
16 models, and a head-to-head between <b>Bagging</b>, <b>Boosting</b>, <b>Stacking</b> and <b>Hybrid</b> ensembles.</p></div>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("🏆 Best model (hybrid)", best.replace("Hybrid ", ""))
k2.metric("Best test MAE", f"{lb.loc[best, 'MAE']:.3f} °C", f"{lb.loc[best, 'MAE'] - pers:+.3f} °C vs persistence", delta_color="inverse")
k3.metric("Skill vs 'tomorrow = today'", f"{lb.loc[best, 'Skill vs persistence (%)']:+.1f} %")
k4.metric("Tree error cut by bagging", f"{(1 - lb.loc['Random Forest', 'MAE'] / lb.loc['Decision Tree', 'MAE']) * 100:.0f} %",
          "Decision Tree → Random Forest")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏝️ Climate story", "🏆 Model arena", "📈 Forecast explorer",
                                         "🧪 Ensemble lab", "🔮 Tomorrow predictor"])

# ---------------- Tab 1: Climate story ----------------
with tab1:
    c1, c2 = st.columns([1, 1.6])
    with c1:
        st.subheader("Three climates, one small island")
        mp = pd.DataFrame([(c, *v) for c, v in CITY_INFO.items()], columns=["city", "lat", "lon", "zone"])
        g_ = wdaily.groupby("city")
        mp["elevation (m)"] = mp.city.map(g_.elevation.first())
        mp["avg max °C"] = mp.city.map(g_.temperature_2m_max.mean().round(1))
        mp["annual rain (mm)"] = mp.city.map((g_.precipitation_sum.mean() * 365.25).round(0))
        fig = px.scatter(mp, x="elevation (m)", y="avg max °C", color="city", text="city", size="annual rain (mm)",
                         size_max=55, color_discrete_map=CITY_COLORS, hover_data=["zone", "annual rain (mm)"])
        fig.update_traces(textposition="top center")
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0), showlegend=False,
                          title="Higher = cooler (bubble size = annual rainfall)", yaxis_range=[15, 34], xaxis_range=[-250, 2200])
        st.plotly_chart(fig, use_container_width=True)
        st.caption(" · ".join(f"**{c}** – {v[2]}" for c, v in CITY_INFO.items()))
    with c2:
        w = wdaily.copy()
        w["m"] = w.date.dt.to_period("M").dt.to_timestamp()
        mm = w.groupby(["city", "m"]).temperature_2m_max.mean().reset_index()
        fig = px.line(mm, x="m", y="temperature_2m_max", color="city", color_discrete_map=CITY_COLORS,
                      labels={"m": "", "temperature_2m_max": "Monthly mean of daily max (°C)"})
        fig.add_vrect(x0=pd.Timestamp(meta["split"]), x1=mm.m.max(), fillcolor="orange", opacity=0.10, line_width=0)
        fig.add_annotation(x=pd.Timestamp(meta["split"]), y=1, yref="paper", text="test period →", showarrow=False,
                           xanchor="left", yanchor="bottom", font=dict(size=11))
        fig.update_layout(height=270, margin=dict(t=30, b=0), title="Daily max temperature, 2016–2026", legend_title="")
        st.plotly_chart(fig, use_container_width=True)
        w["month"] = w.date.dt.month
        rain = (w.groupby(["city", "month"]).precipitation_sum.mean() * 30.4).reset_index()
        fig = px.bar(rain, x="month", y="precipitation_sum", color="city", barmode="group", color_discrete_map=CITY_COLORS,
                     labels={"precipitation_sum": "Avg monthly rain (mm)", "month": "Month"})
        for x0, x1, lab in [(0.5, 2.5, "NE monsoon"), (2.5, 4.5, "1st inter"), (4.5, 9.5, "SW monsoon"),
                            (9.5, 11.5, "2nd inter"), (11.5, 12.5, "NE")]:
            fig.add_vrect(x0=x0, x1=x1, opacity=0.05 if "inter" in lab else 0.0, fillcolor="grey", line_width=0,
                          annotation_text=lab, annotation_position="top left", annotation_font_size=10)
        fig.update_layout(height=270, margin=dict(t=30, b=0), title="Monsoon rainfall signature", legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    st.info("**Why this matters:** heat-index advisories, school & outdoor-work planning, tea-estate operations and "
            "peak electricity demand (air-conditioning) all depend on *tomorrow's* afternoon temperature. "
            "The 2024–2026 test years were the hottest in this record, which makes the forecasting task harder.")

# ---------------- Tab 2: Model arena ----------------
with tab2:
    st.subheader("Leaderboard: test MAE on 2024–2026 (lower is better)")
    r = lb.sort_values("MAE", ascending=True).reset_index().rename(columns={"index": "model"})
    fig = px.bar(r[::-1], x="MAE", y="model", color="family", orientation="h", color_discrete_map=FAM_COLORS,
                 text=r[::-1].MAE.map(lambda v: f"{v:.3f}"), labels={"MAE": "MAE (°C)", "model": ""})
    fig.add_vline(x=pers, line_dash="dash", line_color="grey", annotation_text="persistence")
    fig.update_layout(height=560, margin=dict(t=10), legend_title="Family", xaxis_range=[0.75, r.MAE.max() * 1.05])
    fig.update_yaxes(categoryorder="array", categoryarray=list(r.model[::-1]))
    st.plotly_chart(fig, use_container_width=True)

    fam = lb.groupby("family").MAE.min().sort_values()
    cols = st.columns(len(fam))
    for c, (f, v) in zip(cols, fam.items()):
        c.metric(f"{f} (°C)", f"{v:.3f}", help=f"Best {f} model: {lb[lb.family == f].MAE.idxmin()}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**MAE by city**")
        fig = px.imshow(per_city.round(3), text_auto=True, color_continuous_scale="RdYlGn_r", aspect="auto",
                        labels=dict(color="MAE °C"))
        fig.update_layout(height=430, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**MAE by monsoon season**")
        s2 = season.reset_index().melt(id_vars=season.index.name or "index", var_name="model", value_name="MAE")
        s2.columns = ["season", "model", "MAE"]
        fig = px.bar(s2, x="season", y="MAE", color="model", barmode="group", labels={"MAE": "MAE (°C)", "season": ""})
        fig.update_layout(height=430, margin=dict(t=10), legend=dict(orientation="h", y=-0.2), yaxis_range=[0.6, s2.MAE.max() * 1.05])
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("Full results table"):
        st.dataframe(lb.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}", "fit_s": "{:.1f}",
                                      "Skill vs persistence (%)": "{:+.1f}"}), use_container_width=True)

# ---------------- Tab 3: Forecast explorer ----------------
with tab3:
    st.subheader(f"Actual vs forecast — {city}")
    p = preds[preds.city == city].sort_values("forecast_for")
    dmin, dmax = p.forecast_for.min().date(), p.forecast_for.max().date()
    rng = st.slider("Date range", min_value=dmin, max_value=dmax, value=(max(dmin, dmax - pd.Timedelta(days=150)), dmax))
    q = p[(p.forecast_for.dt.date >= rng[0]) & (p.forecast_for.dt.date <= rng[1])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q.forecast_for, y=q.target, name="Actual", line=dict(color="black", width=2.5)))
    for m in chosen:
        fig.add_trace(go.Scatter(x=q.forecast_for, y=q[m], name=m, line=dict(width=1.4)))
    fig.update_layout(height=430, yaxis_title="Max temperature (°C)", margin=dict(t=10), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)
    if chosen:
        mcols = st.columns(len(chosen))
        for c, m in zip(mcols, chosen):
            c.metric(m, f"{np.abs(q.target - q[m]).mean():.3f} °C", help="MAE in the selected window")
        c1, c2 = st.columns(2)
        err = q.melt(id_vars=["forecast_for"], value_vars=chosen, var_name="model", value_name="pred")
        err["error"] = err.pred.values - np.tile(q.target.values, len(chosen))
        with c1:
            fig = px.histogram(err, x="error", color="model", nbins=60, barmode="overlay", opacity=0.55,
                               labels={"error": "Forecast − actual (°C)"})
            fig.update_layout(height=330, title="Error distribution", margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            sel = chosen[0]
            fig = px.scatter(q, x="target", y=sel, opacity=0.5,
                             labels={"target": "Actual (°C)", sel: f"{sel} forecast (°C)"})
            lo_, hi_ = q.target.min() - 0.5, q.target.max() + 0.5
            fig.add_shape(type="line", x0=lo_, y0=lo_, x1=hi_, y1=hi_, line=dict(dash="dash", color="grey"))
            fig.update_layout(height=330, title=f"{sel}: forecast vs actual", margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)
        biggest = q.assign(err=np.abs(q.target - q[sel])).nlargest(5, "err")[["forecast_for", "target", sel, "err"]]
        st.markdown(f"**Biggest misses of {sel} in this window** (usually sudden rain / cloud-outbreak days):")
        st.dataframe(biggest.rename(columns={"forecast_for": "date", "target": "actual °C", "err": "abs error"}).round(2),
                     hide_index=True, use_container_width=True)

# ---------------- Tab 4: Ensemble lab ----------------
with tab4:
    c = st.columns(4)
    cards = [("🎒 Bagging", "Train many deep trees on bootstrap samples and <b>average</b> them, which <b>reduces variance</b>.",
              f"Decision Tree {lb.loc['Decision Tree','MAE']:.3f} → RF {lb.loc['Random Forest','MAE']:.3f} °C"),
             ("🚀 Boosting", "Add shallow trees <b>one after another</b>, each fixing the previous errors, which <b>reduces bias</b>.",
              f"Best: {lb[lb.family=='Boosting'].MAE.idxmin()} {lb[lb.family=='Boosting'].MAE.min():.3f} °C"),
             ("🧱 Stacking", "A <b>meta-learner</b> learns how to weight diverse base models trained out-of-fold.",
              f"Learned {lb.loc['Stacking (meta-learner)','MAE']:.3f} vs simple average {lb.loc['Simple Average (5 models)','MAE']:.3f} °C"),
             ("🧬 Hybrid", "Mix <b>different model families</b>: linear/ARIMA for the trend, trees for the non-linear residuals.",
              f"Best overall: {best} {lb.loc[best,'MAE']:.3f} °C")]
    for col, (t, d, res_) in zip(c, cards):
        col.markdown(f"<div class='card'><h4>{t}</h4>{d}<br><br><b>{res_}</b></div>", unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(bcurve, x="k", y="mae", markers=True, log_x=True,
                      labels={"k": "Number of trees averaged (log scale)", "mae": "Test MAE (°C)"})
        fig.add_hline(y=lb.loc["Persistence (baseline)", "MAE"], line_dash="dash", annotation_text="persistence")
        fig.update_layout(height=360, title="Bagging in action: averaging more trees lowers the error", margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        sw = pd.Series(meta["stack_weights"]).reset_index()
        sw.columns = ["base model", "weight"]
        fig = px.bar(sw, x="base model", y="weight", color="base model", text=sw.weight.round(2))
        fig.update_layout(height=360, showlegend=False, margin=dict(t=40),
                          title="Stacking meta-learner weights (non-negative linear)")
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Why does the learned stacker drop LightGBM, CatBoost and the ANN? Their predictions are >99.6 % correlated with RF and "
               "Linear Regression, so the meta-learner can't separate them and keeps only two. With base models this similar and a "
               "warming trend, equal weights (simple average) are more robust: the classic *forecast-combination puzzle*.")
    f = fimp.copy()
    f.index.name = "feature"
    f = f.reset_index().melt(id_vars="feature", var_name="method", value_name="importance").dropna()
    f["importance"] = f.groupby("method").importance.transform(lambda s: s / s.max())
    top = f.groupby("feature").importance.max().nlargest(15).index
    fig = px.bar(f[f.feature.isin(top)], x="importance", y="feature", color="method", barmode="group", orientation="h",
                 labels={"importance": "Relative importance (scaled to max = 1)", "feature": ""})
    fig.update_layout(height=520, title="What drives tomorrow's temperature?", yaxis={"categoryorder": "total ascending"}, margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("`tmax_lag0` = today's max · `tmax_change` = change since yesterday · `dtr` = diurnal range (clear vs cloudy sky) · "
               "`rad_lag0` = today's solar radiation · `rain_lag0` = today's rain · `doy_sin` = season")

# ---------------- Tab 5: Tomorrow predictor ----------------
with tab5:
    st.subheader("🔮 What-if: change today's weather, see tomorrow's forecast from each ensemble")
    try:
        M = load_models()
    except Exception as e:  # pragma: no cover
        st.error(f"Could not load models.joblib: {e}")
        st.stop()
    tc = tfeat[tfeat.city == city].sort_values("date")
    c1, c2 = st.columns([1, 2])
    with c1:
        day = st.date_input("Start from a real day (today's observations)", value=tc.date.max().date(),
                            min_value=tc.date.min().date(), max_value=tc.date.max().date())
        row = tc[tc.date.dt.date == day]
        if row.empty:
            row = tc.iloc[[-1]]
        base = row.iloc[0]
        st.caption(f"Observed on {base.date.date()} in {city}")
        tmax = st.slider("Today's max temperature (°C)", 10.0, 40.0, float(round(base.tmax_lag0, 1)), 0.1)
        rain = st.slider("Today's rainfall (mm)", 0.0, 150.0, float(round(base.rain_lag0, 1)), 0.5)
        cloud = st.slider("Today's mean cloud cover (%)", 0.0, 100.0, float(round(base.cloud_lag0)), 1.0)
        rad = st.slider("Today's solar radiation (MJ/m²)", 0.0, 30.0, float(round(base.rad_lag0, 1)), 0.1)
        rh = st.slider("Today's mean humidity (%)", 40.0, 100.0, float(round(base.rh_lag0)), 1.0)
    X = row[meta["features"]].copy().astype(float)
    dT, dR, dC = tmax - base.tmax_lag0, rain - base.rain_lag0, cloud - base.cloud_lag0
    X["tmax_lag0"] = tmax
    X["tmax_change"] += dT
    X["dtr"] += dT
    for w_ in (3, 7, 30):
        X[f"tmax_roll{w_}"] += dT / w_
    X["rain_lag0"], X["rain_sum7"] = rain, X["rain_sum7"] + dR
    X["cloud_lag0"], X["cloud_roll3"] = cloud, X["cloud_roll3"] + dC / 3
    X["rad_lag0"], X["rh_lag0"] = rad, rh
    lag = X["tmax_lag0"].values
    out = {}
    for name in ["Linear Regression", "Random Forest", "LightGBM", "CatBoost", "ANN (MLP)"]:
        out[name] = float(M[name].predict(X)[0] + lag[0])
    out["Stacking (meta-learner)"] = float(M["meta"].predict(pd.DataFrame([out])[list(meta["stack_weights"])])[0])
    out["Hybrid Linear + RF"] = float(M["Linear Regression"].predict(X)[0] + M["Linear+RF residual"].predict(X)[0] + lag[0])
    out["Persistence (baseline)"] = float(tmax)
    actual = preds[(preds.city == city) & (preds.date == base.date)]
    with c2:
        o = pd.Series(out).sort_values().reset_index()
        o.columns = ["model", "forecast"]
        fam_map = {**lb.family.to_dict()}
        o["family"] = o.model.map(fam_map).fillna("Hybrid")
        fig = px.bar(o, x="forecast", y="model", color="family", orientation="h", color_discrete_map=FAM_COLORS,
                     text=o.forecast.map(lambda v: f"{v:.1f} °C"), labels={"forecast": "Forecast max temp tomorrow (°C)", "model": ""})
        if not actual.empty and abs(dT) + abs(dR) + abs(dC) < 1e-6 and abs(rad - base.rad_lag0) < 0.05 and abs(rh - base.rh_lag0) < 0.5:
            fig.add_vline(x=float(actual.target.iloc[0]), line_dash="dash", line_color="black", annotation_text="actual")
        fig.update_layout(height=430, xaxis_range=[min(o.forecast.min(), tmax) - 2, max(o.forecast.max(), tmax) + 1.5], margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
        hyb = out["Hybrid Linear + RF"]
        heat = "🔥 Hot afternoon — stay hydrated" if hyb >= 32 else ("🌤️ Warm" if hyb >= 26 else ("🌥️ Mild" if hyb >= 20 else "🧥 Cool highland day"))
        st.success(f"**{city}, tomorrow ({(base.date + pd.Timedelta(days=1)).date()}):** best-model forecast **{hyb:.1f} °C** · {heat}")
        st.caption("Try it: raise today's rainfall and cloud cover. The tree ensembles react with non-linear jumps, "
                   "while the linear model moves smoothly. That difference is why the hybrid works.")

st.divider()
st.caption("Data: Weather data by Open-Meteo.com (CC BY 4.0), ERA5 reanalysis · Project: Ensemble Learning hands-on · ")
