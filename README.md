# AEMET Weather Forecasting

Multi-horizon forecasting of daily maximum temperature, minimum temperature and precipitation using historical SYNOP station data from AEMET (Spain's State Meteorological Agency).

**Status:** In progress — Data pipeline, EDA and ML baseline complete. DL models upcoming.

---

## Overview

This project builds and evaluates ML/DL models to **forecast daily maximum temperature, minimum temperature and precipitation at 1 to 7-day horizons** across Spain's SYNOP meteorological network.

The forecasting task is framed as a multivariate regression problem over meteorological time series. Classical ML approaches (LightGBM) are evaluated at t+1, t+3 and t+7 as reference baselines. The deep learning phase develops a Bidirectional LSTM architecture with station embedding that predicts all seven daily horizons simultaneously across the SYNOP station network.

## EDA Highlights

### Station network coverage
![Station map](output_figures/map_selected_stations.png)

### Long-term warming trend — decadal evolution of daily maximum temperature
![Decadal evolution tmax](output_figures/decadal_evolution_tmax.png)

### Time series decomposition — daily maximum temperature · Madrid-Retiro
![Decomposition tmax Madrid-Retiro](output_figures/decomposition_3195_tmax.png)

---

## Project Phases

**Phase 1 — EDA & Station Selection** <br>
Exploratory analysis of the full dataset. Selection of stations with 
continuous records since 1991 and less than 10% missing data, following 
WMO quality standards. Spatial and temporal characterisation of temperature 
and precipitation across Spain's SYNOP network.

**Phase 2 — ML Baseline** <br>
Training and evaluation of XGBoost and LightGBM regressors for 1, 3 and 7-day forecast horizons. Random Forest was evaluated and discarded due to systematically higher MAE and training times 8–10× longer than gradient boosting models. Evaluation metrics: RMSE, MAE and R². LightGBM and XGBoost produced comparable results across all prototype stations and horizons; LightGBM was selected as the final ML baseline based on consistent superiority at t+1 and greater stability across stations. A single model is retained as baseline to provide a clean reference point for DL comparison.

**Phase 3 — DL Sequence Models** *(in progress)*  
Development of a LSTM architecture with station embedding for multi-horizon sequence forecasting. The model predicts all seven daily horizons (t+1 to t+7) simultaneously for maximum temperature, minimum temperature and precipitation. The training set is determined by data availability, recent gap tolerance and valid observations in recent months, and varies with model configuration.

---

## Data

| | |
|---|---|
| **Agency** | AEMET — Agencia Estatal de Meteorología |
| **Network** | SYNOP (~98 stations across Spain) |
| **API** | AEMET OpenData API |
| **Target variables** | Daily maximum temperature · Daily minimum temperature · Daily precipitation |
| **Feature variables** | Lagged tmax, tmin, prec · Atmospheric pressure · Relative humidity · Solar radiation · Wind speed · Altitude · Coordinates |
| **Coverage** | 1991–present (stations with continuous records) |
| **Volume** | 3.5M+ daily records (full database); 1.25M filtered records (stations meeting WMO quality criteria, 1991–present) |

---

## Data Pipeline

A robust extraction pipeline was built to extract and validate historical records from the AEMET API prior to modelling:

- Incremental extraction per station with adaptive retry logic and rate 
  limit management
- Gap auditing: detection and quantification of missing intervals
- Historical backfill: automated recovery of missing records with 
  multi-attempt verification
- PostgreSQL database storing 3M+ validated daily records

Pipeline code available in `data_ingestion/` and `data_audit/`.

---

## Station Selection Criteria

Stations are included if they meet all of the following:

- Continuous records since **1991**, covering the current AEMET 
  climatological reference period (1991–2020)
- Minimum record span of **30 years** (WMO standard)
- **Less than 10% missing data** over the full record span, following 
  the completeness threshold used by the WMO for centennial station 
  recognition

---

## Tech Stack

| Component | Technology |
|---|---|
| Data extraction & processing | Python · Pandas · NumPy |
| Database | PostgreSQL |
| ML models | Scikit-learn · XGBoost · LightGBM |
| DL models | TensorFlow · Keras |
| Visualisation | Matplotlib · Seaborn |
| Version control | Git · GitHub |

---

## Author

Helena Alcolea Ruiz · Physicist (BSc + MSc in Complex Systems) · 
Data Scientist · [LinkedIn](#) · [GitHub](#)
