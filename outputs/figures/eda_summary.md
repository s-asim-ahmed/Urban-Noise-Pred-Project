# EDA Summary

- Highest average borough noise: **Southwark** at **74.50 dB(A)**.
- Quietest average borough noise: **Westminster** at **48.76 dB(A)**.
- Loudest land use category: **Commercial** at **74.43 dB(A)**.
- Quietest land use category: **Park** at **48.83 dB(A)**.
- Peak average hour: **18:00** with **71.40 dB(A)**.
- Quietest average hour: **02:00** with **52.33 dB(A)**.
- Noise and traffic volume correlation: **0.85**.
- Noise and temperature correlation: **0.01**.
- Noise and humidity correlation: **-0.01**.

Interpretation:
The synthetic sample behaves like a plausible urban network: noisier commercial corridors remain consistently louder than park or residential contexts, and the strongest temporal peaks align with high-traffic commuting and daytime activity windows. Traffic volume is the most directly associated predictor among the available covariates, which supports using lagged traffic and short rolling windows in the forecasting stage. Weather variables show weaker associations, so they are useful as contextual features rather than primary drivers.
