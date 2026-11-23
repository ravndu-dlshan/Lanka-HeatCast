# Lanka HeatCast: 3-minute video script (voice-over)

Target length: about 2:50 at a relaxed pace (~420 words). Screen-record the dashboard (`streamlit run app.py`) and the notebook.

---

**[0:00–0:20] Hook (Hero banner + KPI cards)**
"Colombo, Kandy, Nuwara Eliya: three cities no more than about 100 km apart, with three completely different climates.
Tomorrow's afternoon temperature decides heat advisories, school sports days, tea-estate work and how hard our power grid
gets pushed. So I asked: *can ensemble learning forecast tomorrow's maximum temperature in Sri Lanka?*"

**[0:20–0:45] Data & story (Climate story tab)**
"I collected eleven years of real daily weather, from 2016 to November 2026, from the Open-Meteo archive: temperature,
rain, humidity, cloud, solar radiation, wind and pressure. That's almost twelve thousand rows. You can see the climate
zones: higher means cooler. You can also see the monsoon signature: the South-West monsoon soaks Colombo, and the
inter-monsoon storms hit everyone. And notice the test period, 2024 to 2026, was the hottest in the record."

**[0:45–1:05] Method (Notebook, feature-engineering cells)**
"In the notebook I built 57 features: lags, rolling averages, pressure changes and monsoon season. I trained on
2016–2023 and tested on almost three unseen years. The models predict the *change* from today, which lets tree models
keep up with a warming climate."

**[1:05–1:45] Ensembles (Ensemble lab tab)**
"Bagging first. A single decision tree is worse than just saying 'tomorrow equals today'. But average hundreds of
bootstrapped trees, and the error drops by about 20 %. That's variance reduction, live on this curve.
Boosting (Gradient Boosting, LightGBM, CatBoost) reduces bias and edges ahead of bagging.
Then stacking: here's an interesting twist. The learned meta-learner actually lost to a simple average, because the base
models were more than 99 % correlated. Forecasters call this the forecast-combination puzzle."

**[1:45–2:15] Results (Model arena tab)**
"So who won? A **hybrid**. Linear regression captures the smooth, persistent signal and can extrapolate to record heat,
and a Random Forest learns the non-linear leftovers, like heavy rain plus thick cloud. That gives a 0.83 °C average error,
10 % better than persistence. The ARIMA-plus-Random-Forest hybrid came close behind. The hill country and the NE monsoon
are the hardest to forecast."

**[2:15–2:45] Demo (Forecast explorer → Tomorrow predictor)**
"Here are forecasts against reality for Colombo. In the what-if predictor I can pick a day and pour on 60 mm of rain and
heavy cloud. Watch the tree ensembles jump while the linear model moves smoothly. That's exactly why combining them works."

**[2:45–3:00] Close**
"Lanka HeatCast: real Sri Lankan data, bagging, boosting, stacking and hybrids, with honest reasons for why each one
behaves the way it does. Code and UI were built with the help of Claude, a generative-AI assistant. Thank you!"
