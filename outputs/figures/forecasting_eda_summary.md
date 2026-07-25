# Forecasting EDA Summary

- Rows analysed: **1994**
- Locations analysed: **50**
- Coverage window: **2023-01-01 03:00:00** to **2023-12-31 20:00:00**
- Peak hour-of-day: **15:00**
- Quietest hour-of-day: **04:00**
- Strongest average day-of-week: **2**
- Weekday minus weekend mean noise gap: **0.83 dB(A)**
- Mean lag-24 autocorrelation: **0.01**
- Noise to traffic correlation: **0.01**
- Noise to temperature correlation: **0.03**
- Noise to wind-speed correlation: **0.03**

Model design note:
The dataset shows strong intra-day structure and traffic-linked variation, so the forecasting feature uses a seasonal SARIMA candidate with a 24-hour cycle and falls back to a simpler ARIMA candidate when the seasonal fit is unstable. This keeps the prediction pathway date-driven and aligned with the hourly resolution of the source data.
