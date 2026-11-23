# 🌦️ Lanka HeatCast: Ensemble Learning Dashboard

Forecasting **tomorrow's maximum temperature** for **Colombo, Kandy and Nuwara Eliya** with
**Bagging, Boosting, Stacking and Hybrid** ensembles, using 11 years of real Sri Lankan weather data.

| | |
|---|---|
| **Data** | Open-Meteo Historical Weather API (ERA5 reanalysis), daily, 2016-01-01 → 2026-11-15, 3 cities × 3,972 days, 9 variables (`data/`) |
| **Target** | Next-day maximum 2 m temperature (°C) |
| **Split** | Train 2016–2023 · Test 2024-01-01 → 2026-11-14 (chronological, 3,147 forecasts) |
| **Bagging** | Bagged Trees, Random Forest, Extra Trees |
| **Boosting** | AdaBoost, Gradient Boosting, LightGBM, CatBoost |
| **Stacking** | Linear + RF + LightGBM + CatBoost + ANN → non-negative linear meta-learner (time-ordered out-of-fold), plus simple average |
| **Hybrids** | ARIMA + Random Forest (Zhang residual hybrid) · RF + ANN blend · Linear Regression + RF/CatBoost residual |
| **Best model** | Hybrid Linear + RF, MAE **0.830 °C**, R² 0.948, +10.1 % skill vs persistence |

## Files
* `Ensemble_Weather_SriLanka.ipynb`: the full, executed pipeline (EDA → features → models → evaluation → interpretation). Running it re-creates `artifacts/`.
* `app.py`: Streamlit dashboard with 5 tabs: Climate story · Model arena · Forecast explorer · Ensemble lab · Tomorrow predictor (what-if).
* `artifacts/`: leaderboard, predictions, feature importances and trained models used by the dashboard.
* `data/sri_lanka_weather_2016_2026.csv`: the dataset. If it is missing, the notebook downloads it again from Open-Meteo.

## Run locally
```bash
pip install -r requirements.txt
jupyter notebook Ensemble_Weather_SriLanka.ipynb   # optional: re-train (≈5 min)
streamlit run app.py
```

## Deploy (optional live link)
1. Push this folder to a public GitHub repository.
2. Go to https://share.streamlit.io, choose **New app**, select the repo, branch `main` and file `app.py`, and set Python to **3.13** under *Advanced settings*.
3. Deploy, then submit the `https://<name>.streamlit.app` URL.



*Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0), based on ERA5 reanalysis (Copernicus Climate Change Service).*
