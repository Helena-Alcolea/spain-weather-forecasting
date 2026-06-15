# Dashboard AEMET — previsión a 7 días

App [Streamlit](https://streamlit.io) que muestra las predicciones del modelo seq2seq
(LSTM encoder-decoder + atención + embedding de estación) en un mapa de España con
selector de estación y gráfico de la previsión a 7 días.

Las predicciones van **precomputadas** en [`predictions.json`](predictions.json): la app
solo lee ese fichero estático, así que **no necesita TensorFlow ni la base de datos** y el
despliegue es ligero y gratuito.

## Ver en local

```bash
# entorno efímero, no toca el .venv del modelo
uv run --with streamlit --with plotly streamlit run dashboard/streamlit_app.py
```

Abre http://localhost:8501.

## Publicar online (Streamlit Community Cloud — gratis, tipo Tableau Public)

1. Sube el repo a GitHub (con `dashboard/` dentro).
2. Entra en https://share.streamlit.io con tu cuenta de GitHub.
3. "New app" → elige el repo y rama, y en *Main file path* pon
   `dashboard/streamlit_app.py`.
4. Streamlit instala las dependencias de [`requirements.txt`](requirements.txt) (solo
   streamlit + pandas + plotly, sin TensorFlow) y publica la app en una URL pública
   `https://<nombre>.streamlit.app`.

## Regenerar las predicciones

Cuando se actualice la base de datos o el modelo, regenera el JSON desde la raíz del
proyecto y cópialo aquí:

```bash
uv run python generate_predictions.py --out dashboard/predictions.json
```

> Las predicciones son del último snapshot, no en tiempo real. El pipeline puede
> automatizarse con un cron diario que ejecute `update_db.py` + `generate_predictions.py`.
