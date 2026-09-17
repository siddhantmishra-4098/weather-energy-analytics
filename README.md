# Weather-Energy Analytics (Germany)

An end-to-end pipeline connecting weather data to the German power grid, built as a portfolio project for weather/NWP-driven energy analytics roles. Part 1 ingests weather, generation, and price data, decodes native NWP files (GRIB2), and surfaces everything in a Streamlit dashboard, including where the forecasts are biased. Part 2 picks up that bias question and answers it quantitatively with a Bayesian model of weather -> generation, reporting the forecast-driven accuracy cost as a credible interval rather than a point estimate.

## Pipeline

```
ingest_weather.py  ──┐
ingest_energy.py   ──┼──> analysis.py ──> dashboard.py (Streamlit)
ingest_price.py    ──┘         │
                                ├──> analyze_temp_bias_by_hour.py
                                └──> analyze_energy_bias_by_hour.py
                                                │
                                                ▼
                              notebooks/generation_model.ipynb
                       (weather -> generation Bayesian model,
                        forecast-error cost in MW and EUR)

decode_grib.py   (standalone: ECMWF GRIB2 -> cfgrib/ecCodes -> plot)
```

---

## Part 1: data pipeline and dashboard

### 1. Ingest

- **`src/ingest_weather.py`** pulls hourly historical (ERA5-based) actuals and hourly forecast temperature, wind speed, and cloud cover for Dortmund, NRW from the Open-Meteo API. No API key needed. Forecast data comes from Open-Meteo's **Historical Forecast API**, which archives each forecast run as it stood at the time -- the plain forecast endpoint's `past_days` parameter is continuously rewritten with hindsight and would make forecasts look artificially accurate. Writes `data/raw/weather_historical.parquet` and `data/raw/weather_forecast.parquet`.
- **`src/ingest_energy.py`** pulls hourly German grid load and wind/solar generation, both actuals and Day-Ahead forecasts. Defaults to SMARD (Bundesnetzagentur, no key required); switches to the ENTSO-E Transparency Platform automatically if `ENTSOE_API_TOKEN` is set in `.env`. Writes `data/raw/energy_actuals.parquet` and `data/raw/energy_forecast.parquet`.
- **`src/ingest_price.py`** pulls the hourly German day-ahead electricity price (DE-LU bidding zone), same window and provider pattern as `ingest_energy.py`. Writes `data/raw/energy_prices.parquet`.

Each ingest script fails loudly (non-zero exit, clear error) if its API call fails -- there is no silent fallback to fake data.

### 2. Decode native NWP format

**`src/decode_grib.py`** downloads a real IFS forecast field (2m temperature, analysis + 24h lead time) in native GRIB2 format from ECMWF's open-data feed, decodes it with cfgrib/ecCodes via xarray, crops it to Central Europe, and saves plots. This is the part of the project that specifically exercises NWP binary formats rather than a JSON/REST API:

![ECMWF 2m temperature field, analysis step, decoded from GRIB2 and cropped to Central Europe](docs/images/grib_2t_step00.png)

### 3. Join and score

**`src/analysis.py`** joins weather and energy on timestamp, computes correlation between weather and energy variables, and scores forecast accuracy (MAE, RMSE, bias) of Open-Meteo's forecast weather against observed weather. If `data/raw/energy_prices.parquet` exists, it also joins day-ahead price onto the table and computes `residual_load_mw = total_load_mw - wind_total_mw - solar_mw`, correlated against price. This step is what turns four separate parquet files into the single joined table the dashboard and the bias scripts both read from.

### 4. Dashboard

**`dashboard.py`** is a Streamlit app (Plotly charts) reading `analysis.py`'s output. Four tabs, screenshotted from a live run below.

**Weather vs. Energy** -- overlays a chosen weather variable against a chosen grid variable over the full window, so you can eyeball a relationship (e.g. temperature vs. load) before quantifying it:

![Dashboard: weather vs. energy time series, temperature vs. total grid load](docs/images/dashboard_weather_vs_energy.png)

**Forecast Accuracy** -- Open-Meteo's forecast weather plotted against the historical/reanalysis actual, plus the MAE/RMSE/bias table for all three weather variables. This is where the dashboard stops being descriptive and starts being diagnostic: forecast temperature tracks the actual curve closely in shape but runs high on most peaks:

![Dashboard: Open-Meteo forecast vs. actual temperature, with the MAE/RMSE/bias table](docs/images/dashboard_forecast_accuracy.png)

**Correlation Heatmap** -- Pearson correlation between the three weather variables and five energy variables, over the full joined window:

![Dashboard: correlation heatmap, weather variables against energy variables](docs/images/dashboard_correlation_heatmap.png)

Reading the actual numbers (`data/processed/correlation_matrix.csv`): wind speed correlates with wind generation at r = 0.64 (onshore) and r = 0.62 (total) -- the strongest relationship in the table, as expected from the physics. Temperature correlates with solar generation at r = 0.51, weaker than wind's relationship, since solar output depends on irradiance and cloud cover more directly than on temperature itself; temperature is really standing in for "summer vs. winter." Cloud cover barely correlates with solar (r = 0.00 against `solar_mw` directly) -- surprising on its face, but it's a single-station cloud-cover reading standing in for irradiance across the whole German solar fleet, so a lot of the real relationship is averaged out at national scale.

**Energy Forecast Accuracy** -- the same forecast-vs-actual treatment applied to SMARD/ENTSO-E's own Day-Ahead energy forecasts, not just the weather feeding them:

![Dashboard: Day-Ahead energy forecast vs. actual, total grid load](docs/images/dashboard_energy_forecast_accuracy.png)

### 5. Where the forecasts are biased

The dashboard's accuracy tables give one number per variable (MAE, RMSE, bias). That hides whether the error is spread evenly across the day or concentrated at particular hours -- which matters if you're trying to explain *why* a forecast misses, not just *that* it misses. `analyze_temp_bias_by_hour.py` and `analyze_energy_bias_by_hour.py` exist specifically to break the single accuracy number down by hour of day.

**Temperature forecast bias by hour** (`forecast - actual`, °C):

![Open-Meteo temperature forecast bias by hour of day -- positive (too warm) at every hour, worst overnight, smallest near midday](docs/images/temp_bias_by_hour.png)

The bias is positive at every single hour -- Open-Meteo's forecast runs warm across the board for this window, not just occasionally. It's smallest around 09:00-11:00 UTC (+0.05 to +0.11°C) and largest overnight and around dawn, 00:00-04:00 and 20:00-23:00 UTC (+2.0 to +2.3°C). That's consistent with a model that underestimates nighttime radiative cooling more than it misjudges daytime peak heating.

**Day-Ahead energy forecast bias by hour** (`forecast - actual`, MW):

![Day-Ahead energy forecast bias by hour of day, five variables x 24 hours, red = forecast too high, blue = forecast too low](docs/images/energy_bias_by_hour.png)

Total load is the clearest pattern: forecasts run low overnight (as much as -890 MW around 01:00 UTC) and high through the daytime ramp (+890 to +944 MW around 08:00-13:00 UTC) -- the Day-Ahead forecast underestimates how far load drops overnight and overestimates the daytime plateau. Wind onshore has its single worst hour at 18:00 UTC (+765 MW, forecast too high), lining up with the evening transition the temperature bias plot also flags as error-prone. Solar's bias is small in absolute MW next to load, but it is systematically negative through the middle of the day (-134 to -389 MW, 07:00-15:00 UTC) -- the forecast underestimates midday solar output more often than it overestimates it.

### Transition: from "where is the error" to "what does the error cost"

Everything above answers *where* the weather and energy forecasts are biased. It doesn't answer a question a grid operator or trader actually cares about: if I'm predicting generation from weather, and I only have forecast weather (not the actual weather, which doesn't exist yet), how much worse does that make my generation prediction, in MW and in money? Answering that needs a model that maps weather to generation in the first place -- which is what Part 2 builds, and it needs that model's predictions to come with honest uncertainty, since 90 days of hourly data is not enough to pin the forecast-error contribution down to a single number. That's why Part 2 is Bayesian rather than a quick `sklearn.LinearRegression` fit.

---

## Part 2: Bayesian weather -> generation model

`notebooks/generation_model.ipynb` fits two regressions -- `wind_onshore_mw ~ wind_speed_kmh` and `solar_mw ~ temperature_c + cloud_cover_pct` -- with PyMC/NUTS, then answers the cost question above by evaluating each model twice: once fed the actual weather used to fit it (the error floor), once fed Open-Meteo's forecast weather for the same hours (what you'd actually get day-to-day). The gap between the two, isolated per posterior draw, is the accuracy cost attributable specifically to weather forecast error. The notebook follows the Bayesian workflow in order, and each step exists because of what the previous one found -- not as a checklist.

**1. Load data.** Reads `weather_energy_joined.parquet` (from `src/analysis.py`), `weather_forecast.parquet`, and, optionally, `energy_prices.parquet` for the price-weighted cost at the end. Fails loudly if the required files are missing.

**2. EDA on the marginal distributions, before any modeling.** Before picking a likelihood, look at the raw shape of each response variable:

![Marginal histograms: wind_onshore_mw (right-skewed, never zero) and solar_mw (large mass at exactly zero, right-skewed positive tail)](docs/images/generation_model_eda_marginals.png)

This is the step that decides everything downstream. `wind_onshore_mw` is strictly positive and right-skewed -- it never touches zero in this data. `solar_mw` is a genuinely two-part shape: a large spike at exactly 0 MW (nighttime hours) plus a separate right-skewed continuous process for daylight hours. A Normal likelihood fit to either of these can and does predict physically impossible negative generation in its tails, and it has no way to represent solar's exact-zero spike at all.

*Transition: EDA -> likelihood choice.* Because the EDA showed two different shapes, the notebook picks two different likelihoods rather than defaulting to one: wind gets a **Gamma GLM with a log link** (strictly positive, right-skewed, log link keeps the mean positive for any linear predictor), and solar gets a **hurdle-Gamma** -- a Bernoulli gate for zero-vs-nonzero times a Gamma for the positive part conditional on being nonzero. This is a hurdle, not zero-inflation: a continuous Gamma already assigns exactly zero probability to the point y=0, so every nighttime zero has to come from a separate gate mechanism, not from the Gamma itself.

*Transition: likelihood choice -> priors.* Choosing a likelihood family fixes the shape but not the scale -- a `Normal(0, 1)` prior on an intercept means something very different once the response has moved from a z-scored abstraction to raw MW on a log-mu scale. Priors are set with explicit scale-awareness: intercepts are centered at the log of each response's typical order of magnitude, and the Gamma shape parameter's prior is checked for pathological mass near shape~=0 rather than left at a diffuse textbook default.

**3. Prior predictive check.** Before the model ever sees the data, simulate from the prior alone and check the simulated values land in a physically plausible range (thousands to tens of thousands of MW, not negative, not absurdly large). This is what actually validates the scale-aware priors chosen in the previous step, rather than just asserting they're reasonable.

*Transition: prior predictive check -> fit.* Once the priors are shown to produce sane pre-data predictions, it's safe to let the data update them.

**4. Fit.** Both models sampled with PyMC's NUTS sampler.

**5. Convergence diagnostics, then trace plots.** R-hat and effective sample size are checked numerically first; trace plots are the same check made visual, since a chain that's technically within the R-hat threshold can still show a visually obvious problem a summary statistic misses.

*Transition: diagnostics -> posterior predictive check.* Convergence diagnostics confirm NUTS explored the posterior properly -- they say nothing about whether the *model itself* fits the data.

**6. Posterior predictive check.** Draw from the fitted model's posterior predictive distribution and compare it back to the actual observed values:

![Posterior predictive check: simulated draws from the fitted Gamma (wind) and hurdle-Gamma (solar) models against the observed data](docs/images/generation_model_ppc.png)

*Transition: posterior predictive check -> formal comparison.* A PPC is a visual, single-model check. The natural next question is comparative: is this actually better than the naive model a less careful version of this analysis would have shipped?

**7. Formal model comparison: PSIS-LOO vs. a Normal-likelihood baseline.** The Normal baseline exists here *only* to be discredited by this comparison -- it isn't diagnosed on its own merits, since the EDA and PPC steps already showed why it's the wrong likelihood for both targets. `az.compare` backs that up numerically: for wind, the Gamma model beats the Normal baseline by an ELPD difference of 273 (SE 34.9); for solar, hurdle-Gamma beats it by 6,007 (SE 210.2) -- an order of magnitude larger gap, matching how much more badly a Normal likelihood misrepresents solar's zero-spike-plus-right-skew shape compared to wind's simpler right-skew.

**8. Best-case vs. forecast-driven MAE gap, and price-weighted cost.** This is where Part 1's forecast-bias question finally gets a dollar-and-MW answer. Every posterior draw predicts twice -- once on actual weather, once on forecast weather -- so the gap between the two comes out as a full distribution, not a point estimate:

![Best-case (actual weather) vs. forecast-driven (forecast weather) MAE, with 94% credible-interval error bars, both models](docs/images/generation_model_mae.png)

From the current run (`data/processed/generation_model_summary.txt`): the forecast-driven accuracy cost is **+456 MW (94% CI: 414, 500)** for wind and **+1,587 MW (94% CI: 1,317, 1,882)** for solar. Solar's gap is over three times wider than wind's, which tracks back through every earlier step: solar's hurdle structure has more parameters to get wrong from imperfect weather (both the zero/nonzero gate and the positive-part mean depend on forecast temperature and cloud cover), and cloud cover, from the dashboard's own correlation heatmap, is the weakest-correlated predictor in the whole table. Price-weighted against the actual hourly day-ahead price, this comes out to an illustrative imbalance-equivalent cost of EUR 1.21B (wind) and EUR 3.74B (solar) over the ~2,100 priced hours in this window -- clearly **not** a real settlement estimate (real imbalance settlement uses signed error against a dedicated imbalance price, not absolute error against the day-ahead price), but a EUR-scale sense of how much a 90-day window's worth of forecast error is worth at real German price levels.

**9. Save results.** Coefficients, diagnostics, LOO comparison tables, the MAE gap, and the price-weighted cost are all written to `data/processed/generation_model_*.csv` / `.txt`, plus every plot above, as PNGs. All of it is gitignored -- it's produced by running the notebook, not shipped in the repo.

---

## Tools used

Python 3.11, `requests`, pandas/pyarrow, xarray + cfgrib (ecCodes), `ecmwf-opendata`, `entsoe-py`, scikit-learn, PyMC + ArviZ (Bayesian regression, NUTS diagnostics, PSIS-LOO), Open-Meteo API, ENTSO-E Transparency Platform / SMARD API, ECMWF open-data (GRIB2), Plotly, Streamlit, matplotlib.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash; .venv\Scripts\activate.bat on cmd
pip install -r requirements.txt
cp .env.example .env               # edit .env if using ENTSO-E
```

No API key is required to run the pipeline as-is. See `.env.example` for how to request a free ENTSO-E token, and for the location settings (defaults to Dortmund, NRW).

## Running the pipeline

```bash
python src/ingest_weather.py            # -> data/raw/weather_historical.parquet, weather_forecast.parquet
python src/ingest_energy.py             # -> data/raw/energy_actuals.parquet, energy_forecast.parquet
python src/ingest_price.py              # -> data/raw/energy_prices.parquet
python src/decode_grib.py               # -> data/processed/grib_2t_step*.png
python src/analysis.py                  # -> data/processed/weather_energy_joined.parquet, *.csv, weather_energy_price_joined.parquet
python src/analyze_temp_bias_by_hour.py    # -> data/processed/temp_bias_by_hour.png
python src/analyze_energy_bias_by_hour.py  # -> data/processed/energy_bias_by_hour.png
jupyter notebook notebooks/generation_model.ipynb   # -> data/processed/generation_model_*.csv/.txt/.png (PyMC/NUTS)
streamlit run dashboard.py              # opens the dashboard in your browser
```

Each ingest script fails loudly (non-zero exit, clear error message) if its API call fails. There is no silent fallback to fake or mocked data.

## Design notes and known limitations

- **Historical window is capped at 90 days**, so the pipeline runs fast and stays well within API rate limits.
- **Correlation and the generation model use a single weather station (Dortmund) against nationwide German totals.** A production system would use a generation-weighted spatial average across German wind/solar sites. The physical relationships still show up at national scale because German synoptic weather is spatially correlated over hundreds of kilometers, but a single station is a real simplification, not just a caveat for show -- it's the most likely explanation for why cloud cover barely correlates with solar generation in the heatmap above.
- **The imbalance-cost figure is price-weighted by the actual hourly day-ahead price** (`data/raw/energy_prices.parquet`, from `src/ingest_price.py`), not a flat EUR/MWh placeholder -- but it is still not a real settlement price: real imbalance settlement uses the *signed* forecast error against a dedicated imbalance/balancing price for each settlement period, not the absolute forecast-driven error against the day-ahead price used here.
- **Forecast accuracy for weather compares Open-Meteo's own archived forecast against its own historical/reanalysis-backed actuals**, both from the same provider. This measures realistic forecast skill but is not a cross-provider comparison. Comparing against ECMWF HRES (already partly wired up via `decode_grib.py`) would be a natural extension.
- **ECMWF open-data has no server-side spatial subsetting** on the free feed, so `decode_grib.py` downloads the full global field for one parameter/step and crops it client-side.
- **The ENTSO-E day-ahead price query needs a different area code than load/generation.** `query_day_ahead_prices` raises `NoMatchingDataError` against the same `ENTSOE_AREA_CODE` (`10Y1001A1001A83F`) that `query_load`/`query_generation` accept -- it needs the actual DE-LU bidding-zone EIC code, `10Y1001A1001A82H` (`ENTSOE_PRICE_AREA_CODE` in `.env.example`).
- **PyMC's compiled (C) backend may not work out of the box on every machine.** On a venv built from the Windows Store Python package, PyTensor's C linker fails to link against that install's restricted package directory; the notebook works around this with `PYTENSOR_FLAGS=cxx=` (pure-Python linker -- correct, just slower). If you're on a normal Python install where compilation works, delete that line for a large speedup.

## Project structure

```
weather-energy-analytics/
├── data/
│   ├── raw/            # gitignored, populated by ingest_*.py
│   └── processed/      # gitignored, populated by analysis.py / decode_grib.py / notebooks
├── docs/
│   └── images/          # dashboard screenshots and notebook plots referenced by this README
├── notebooks/
│   └── generation_model.ipynb
├── src/
│   ├── ingest_weather.py
│   ├── ingest_energy.py
│   ├── ingest_price.py
│   ├── decode_grib.py
│   ├── analysis.py
│   ├── analyze_temp_bias_by_hour.py
│   └── analyze_energy_bias_by_hour.py
├── dashboard.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```
