# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/putri43/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/putri43/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Unlock-wave ensemble forecaster: forecasts Twitch viewers for GTA VI's launch day (hours 2–24, five quantiles each) from the first hour's print, using the release-day curves of past big game launches shipped in each case as comparables. A launch day is an unlock wave — hour 1 is mostly New Zealand / early Europe, the US unlocks lift the day, and the evening peak is several times hour 1 for a clean midnight launch, while a launch whose category was already huge before release peaks in hour 1 and decays. The forecast is the log-average of three per-hour models (ridge on hour-1 size + pre-release level + channels; local-linear on hour-1 size; smoothed local-linear), each calibrated on the comparables inside the case. Local validation on the four practice cases: mean pinball loss on the log scale ~0.159, vs the house baseline's ~0.387.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=t70w2dhy&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
