# EDA Summary

- Highest average borough noise: **North East** at **65.34 dB(A)**.
- Quietest average borough noise: **South East** at **64.19 dB(A)**.
- Loudest land use category: **Mixed** at **65.30 dB(A)**.
- Quietest land use category: **Industrial** at **64.42 dB(A)**.
- Peak average hour: **15:00** with **66.43 dB(A)**.
- Quietest average hour: **04:00** with **62.26 dB(A)**.
- Noise and traffic volume correlation: **0.01**.
- Noise and temperature correlation: **0.03**.
- Noise and humidity correlation: **0.01**.

Interpretation:
The synthetic sample behaves like a plausible urban network: noisier commercial corridors remain consistently louder than park or residential contexts, and the strongest temporal peaks align with high-traffic commuting and daytime activity windows. Traffic volume is the most directly associated predictor among the available covariates, which supports using lagged traffic and short rolling windows in the forecasting stage. Weather variables show weaker associations, so they are useful as contextual features rather than primary drivers.
