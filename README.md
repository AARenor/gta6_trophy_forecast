# gta6_trophy_forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/AARenor/gta6_trophy_forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/AARenor/gta6_trophy_forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each Grand Theft Auto VI PlayStation trophy's earn rate on Exophase. Earners follow a two-component completion curve — a dedicated week-one wave plus a slow casual tail — convolved with growing players-tracked, so late joiners dilute the rate. Story trophies share a pace fit from steps that have already moved, then that curve is shifted along the mission order for the steps behind them. Saturation ceilings are lifetime priors from GTA V, Red Dead Redemption 2, Cyberpunk 2077 and Elden Ring on Exophase; the pace is fit from the case. Spreads are logit errors from a holdout of the case's own history, scaled out to the target horizon.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
