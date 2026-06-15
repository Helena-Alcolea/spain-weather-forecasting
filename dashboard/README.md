# Weather Forecast Dashboard — Spain

A [Streamlit](https://streamlit.io) app that shows the seq2seq model's predictions
(LSTM encoder–decoder + attention + per-station embedding) on a map of Spain, with a
station selector and the 7-day forecast charts.

Predictions are **precomputed** in [`predictions.json`](predictions.json): the app only
reads that static file, so it needs **neither TensorFlow nor the database** and the
deployment is lightweight and free.

## Run locally

```bash
# ephemeral environment, does not touch the model's .venv
uv run --with streamlit --with plotly streamlit run dashboard/streamlit_app.py
```

Open http://localhost:8501.

## Publish online (Streamlit Community Cloud — free, Tableau-Public style)

1. Push the repo to GitHub (with `dashboard/` inside).
2. Sign in to https://share.streamlit.io with your GitHub account.
3. "New app" → pick the repo and branch, and set *Main file path* to
   `dashboard/streamlit_app.py`.
4. Streamlit installs the dependencies from [`requirements.txt`](requirements.txt) (just
   streamlit + pandas + plotly, no TensorFlow) and publishes the app at a public URL
   `https://<name>.streamlit.app`.

## Regenerate the predictions

When the database or the model is updated, regenerate the JSON from the project root and
copy it here:

```bash
uv run python generate_predictions.py --out dashboard/predictions.json
```

> Predictions reflect the latest snapshot, not real time. The pipeline can be automated
> with a daily cron job running `update_db.py` + `generate_predictions.py`.

---

*Data source: AEMET open data. Independent portfolio project; not affiliated with AEMET.*
