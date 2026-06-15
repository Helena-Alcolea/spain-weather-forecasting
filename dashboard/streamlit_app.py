"""Dashboard AEMET — previsión a 7 días por estación.

App Streamlit que lee `predictions.json` (predicciones precomputadas por el modelo) y
las muestra en un mapa de España con selector de estación y los gráficos de la previsión
a 7 días. No necesita TensorFlow ni base de datos: solo lee el JSON, por eso el despliegue
en Streamlit Community Cloud es ligero y gratuito.

Local:   uv run --with streamlit --with plotly streamlit run dashboard/streamlit_app.py
Online:  Streamlit Community Cloud apuntando a este fichero (usa dashboard/requirements.txt).
"""

from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "predictions.json"

# Paleta cielo / naturaleza (para la barra de título y el mapa).
SKY_BLUE = "rgb(0, 141, 218)"     # azul cielo intenso
CYAN = "rgb(65, 201, 226)"        # cian claro
AQUA = "rgb(172, 226, 225)"       # aguamarina pálido
CREAM = "rgb(247, 238, 221)"      # crema (punto medio del mapa)
BG = "#EFF2F7"                    # fondo general: gris-azulado muy claro rgb(239,242,247)
BOX_BG = "#FFFFFF"                # fondo de recuadros y gráficas: blanco
BOX_BORDER = "rgb(45, 87, 99)"    # borde fino de los recuadros
# Temperaturas: convención estándar (cálido = máxima, frío = mínima).
TMAX_COLOR = "#d62728"            # rojo
TMIN_COLOR = "#1f77b4"            # azul
PRECIP_COLOR = SKY_BLUE

# Traducciones de toda la interfaz (los nombres de estación/provincia son de AEMET).
TR = {
    "es": {
        "title": "🌦️ Previsión meteorológica a 7 días — España",
        "select": "Selecciona una estación (o pincha en el mapa)",
        "colorbar": "Tª media °C",
        "hover_mean": "Tª media",
        "forecast_for": "Previsión para",
        "elevation": "Altitud",
        "mean_title": "Temperatura media (°C)",
        "temp_title": "Temperaturas máximas y mínimas (°C)",
        "legend_max": "máxima",
        "legend_min": "mínima",
        "precip_title": "Precipitación (mm)",
        "precip_caption": "Lluvia acumulada prevista por día",
        "past": "Días pasados",
        "future": "Días futuros",
        "footer": "Fuente de datos: AEMET · Dashboard por Helena Alcolea Ruiz",
        "dow": ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"],
        "mes": ["ene", "feb", "mar", "abr", "may", "jun",
                "jul", "ago", "sep", "oct", "nov", "dic"],
    },
    "en": {
        "title": "🌦️ 7-day weather forecast — Spain",
        "select": "Choose a station (or click on the map)",
        "colorbar": "Mean temp °C",
        "hover_mean": "Mean temp",
        "forecast_for": "Forecast for",
        "elevation": "Elevation",
        "mean_title": "Mean temperature (°C)",
        "temp_title": "Maximum and minimum temperatures (°C)",
        "legend_max": "max",
        "legend_min": "min",
        "precip_title": "Precipitation (mm)",
        "precip_caption": "Forecast daily accumulated rainfall",
        "past": "Past days",
        "future": "Future days",
        "footer": "Data source: AEMET · Dashboard by Helena Alcolea Ruiz",
        "dow": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
        "mes": ["jan", "feb", "mar", "apr", "may", "jun",
                "jul", "aug", "sep", "oct", "nov", "dec"],
    },
}

st.set_page_config(page_title="Spain · 7-day weather forecast", page_icon="🌦️", layout="wide")


def inject_style() -> None:
    """Fondo casi blanco + barra de título con degradado de la paleta."""
    st.markdown(
        f"""
        <style>
          .stApp {{ background-color: {BG}; }}
          .titlebar {{
              background: linear-gradient(90deg, {SKY_BLUE}, {CYAN});
              padding: 18px 28px; border-radius: 14px; margin-bottom: 20px;
              box-shadow: 0 2px 8px rgba(0,0,0,0.08);
          }}
          .titlebar h1 {{ color: white; margin: 0; font-size: 1.7rem; font-weight: 700; }}
          .blocktitle {{ font-size: 1.35rem; font-weight: 700; color: #2c3e50; margin-bottom: 8px; }}
          .meanstrip {{ display: flex; justify-content: space-between; gap: 6px; margin-bottom: 2px; }}
          .meanitem {{ flex: 1; text-align: center; }}
          .meanitem .mt {{ font-size: 1.15rem; font-weight: 600; color: #2c3e50; }}
          .meanitem .md {{ font-size: 0.92rem; color: #3a4a57; }}
          .meanitem .mw {{ font-size: 0.82rem; color: #5a6b7b; }}
          /* Recuadro de cada bloque: fondo blanco en TODO el recuadro + borde fino.
             Se apunta por la clase st-key-<key> del container (fiable) y por el testid. */
          div[data-testid="stVerticalBlockBorderWrapper"],
          div[class*="st-key-card-"] {{
              border: 1px solid {BOX_BORDER} !important;
              background-color: {BOX_BG} !important;
              border-radius: 10px; padding: 6px 16px 24px !important;
          }}
          /* El selector va sin borde (solo fondo blanco). */
          div[class*="st-key-card-sel"] {{ border: none !important; box-shadow: none !important; }}
          .footer {{ font-size: 0.78rem; color: #8a96a3; text-align: center; margin-top: 22px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data() -> dict:
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


def stations_frame(payload: dict) -> pd.DataFrame:
    """Una fila por estación con la temperatura media del primer día (para el mapa)."""
    rows = []
    for s in payload["estaciones"]:
        d0 = s["dias"][0]
        rows.append({
            "estacion": s["estacion"],
            "nombre": s["nombre"] or s["estacion"],
            "provincia": s["provincia"] or "",
            "lat": s["lat"],
            "lon": s["lon"],
            "altitud": s["altitud"],
            "tmedia_manana": round((d0["tmax"] + d0["tmin"]) / 2, 1),
        })
    return pd.DataFrame(rows)


def forecast_frame(station: dict) -> pd.DataFrame:
    df = pd.DataFrame(station["dias"])
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["tmedia"] = (df["tmax"] + df["tmin"]) / 2
    return df


def mean_block_html(df: pd.DataFrame, t: dict) -> str:
    """Bloque 'Temperatura media': por día, media / fecha / día de la semana."""
    items = "".join(
        f'<div class="meanitem"><div class="mt">{row.tmedia:.0f} °C</div>'
        f'<div class="md">{row.fecha.day} {t["mes"][row.fecha.month - 1]}</div>'
        f'<div class="mw">{t["dow"][row.fecha.weekday()]}</div></div>'
        for row in df.itertuples(index=False)
    )
    return (
        f'<div class="blocktitle">{t["mean_title"]}</div>'
        f'<div class="meanstrip">{items}</div>'
    )


def _date_ticks(fechas: pd.Series, t: dict) -> dict:
    """Eje X con la fecha y, debajo, el día de la semana en el idioma activo."""
    ticktext = [f"{d.day} {t['mes'][d.month - 1]}<br>{t['dow'][d.weekday()]}" for d in fechas]
    return dict(tickmode="array", tickvals=list(fechas), ticktext=ticktext,
                tickfont=dict(size=12, color="#3a4a57"))


def _transparent(fig: go.Figure) -> go.Figure:
    """Fondo blanco explícito de la gráfica (independiente del fondo del recuadro)."""
    fig.update_layout(paper_bgcolor=BOX_BG, plot_bgcolor=BOX_BG)
    fig.update_xaxes(gridcolor="rgba(0,0,0,0.06)")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.06)")
    return fig


def add_today_split(fig: go.Figure, fechas: pd.Series, t: dict, label_y: float = 1.06) -> None:
    """Separa pasado/futuro: línea discontinua antes de hoy + sombreado del futuro.

    Por el desfase de ~4 días en la publicación de AEMET, los primeros días de la
    previsión caen en el pasado (reconstrucción) y solo los últimos son futuro real.
    Se sombrea el FUTURO (convención de dashboards de previsión: destaca lo accionable).
    """
    today = pd.Timestamp.now().normalize()
    dmin, dmax = fechas.min(), fechas.max()
    half = pd.Timedelta(hours=12)
    has_past = bool((fechas < today).any())
    has_future = bool((fechas >= today).any())
    if not (has_past and has_future):
        return  # todo pasado o todo futuro: no hay frontera que dibujar

    boundary = today - half
    fig.add_vrect(x0=boundary, x1=dmax + half, fillcolor="rgba(0,141,218,0.07)",
                  line_width=0, layer="below")
    fig.add_vline(x=boundary, line=dict(color="#9aa7b3", width=1.5, dash="dash"))
    past_center = (dmin - half) + ((boundary - (dmin - half)) / 2)
    fut_center = boundary + (((dmax + half) - boundary) / 2)
    for x, label in [(past_center, t["past"]), (fut_center, t["future"])]:
        fig.add_annotation(x=x, y=label_y, yref="paper", yanchor="bottom", showarrow=False,
                           text=f"<b>{label}</b>", font=dict(size=13, color="#5a6b7b"))


def map_figure(df: pd.DataFrame, selected: str | None, t: dict) -> go.Figure:
    """Mapa de estaciones coloreado por la temperatura media del primer día."""
    fig = go.Figure(go.Scattermap(
        lat=df["lat"], lon=df["lon"], mode="markers",
        marker=dict(
            size=[18 if e == selected else 10 for e in df["estacion"]],
            color=df["tmedia_manana"],
            colorscale=[[0.0, SKY_BLUE], [0.5, CREAM], [1.0, TMAX_COLOR]],
            colorbar=dict(title=t["colorbar"]), opacity=0.92,
        ),
        customdata=df[["estacion", "provincia", "tmedia_manana"]],
        text=df["nombre"],
        hovertemplate=(
            "<b>%{text}</b><br>%{customdata[1]}<br>"
            + t["hover_mean"] + ": %{customdata[2]:.1f}°C<extra></extra>"
        ),
    ))
    fig.update_layout(
        map=dict(style="carto-positron", center=dict(lat=40.0, lon=-3.5), zoom=4.4),
        margin=dict(l=0, r=0, t=0, b=0), height=460,
    )
    return fig


def temp_figure(df: pd.DataFrame, t: dict) -> go.Figure:
    """tmax/tmin como líneas, con el valor en negrita sobre cada punto y leyenda arriba."""
    fig = go.Figure()
    fig.add_scatter(x=df["fecha"], y=df["tmin"], name=t["legend_min"],
                    mode="lines+markers+text", line=dict(color=TMIN_COLOR, width=3),
                    marker=dict(size=8), text=[f"<b>{v:.0f}°</b>" for v in df["tmin"]],
                    textposition="top center", textfont=dict(size=12, color="#2c3e50"))
    fig.add_scatter(x=df["fecha"], y=df["tmax"], name=t["legend_max"],
                    mode="lines+markers+text", line=dict(color=TMAX_COLOR, width=3),
                    marker=dict(size=8), text=[f"<b>{v:.0f}°</b>" for v in df["tmax"]],
                    textposition="top center", textfont=dict(size=12, color="#2c3e50"))
    add_today_split(fig, df["fecha"], t, label_y=1.16)
    # Margen en el eje Y para que el número sobre el punto de la tmáx no se corte arriba.
    ymin, ymax = float(df["tmin"].min()) - 2, float(df["tmax"].max()) + 3
    fig.update_layout(
        template="plotly_white", height=400, margin=dict(l=10, r=10, t=88, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
        xaxis=_date_ticks(df["fecha"], t),
        yaxis=dict(ticksuffix=" °C", range=[ymin, ymax]), hovermode="x unified",
    )
    return _transparent(fig)


def precip_figure(df: pd.DataFrame, t: dict) -> go.Figure:
    """Precipitación prevista en barras, con eje de escala con suelo mínimo.

    El suelo (al menos 8 mm) evita que lluvias de 0.1-0.2 mm aparezcan como barras
    grandes y engañosas: una llovizna debe verse pequeña.
    """
    top = max(8.0, math.ceil(df["prec"].max() * 1.3))
    # Número (en negrita) encima de cada barra, incluido "0.0 mm" en días sin lluvia.
    labels = [f"<b>{v:.1f} mm</b>" for v in df["prec"]]
    fig = go.Figure(go.Bar(
        x=df["fecha"], y=df["prec"], marker_color=PRECIP_COLOR,
        text=labels, textposition="outside", textfont=dict(size=12), cliponaxis=False,
        hovertemplate="%{y:.1f} mm<extra></extra>",
    ))
    add_today_split(fig, df["fecha"], t, label_y=1.05)
    # Altura: el borde inferior del bloque de precipitación cuadra con el del bloque
    # temp máx/mín. El margen inferior (b) centra algo más el gráfico dentro del recuadro.
    fig.update_layout(
        template="plotly_white", height=508, margin=dict(l=10, r=10, t=60, b=30),
        xaxis=_date_ticks(df["fecha"], t),
        yaxis=dict(range=[0, top], ticksuffix=" mm"), hovermode="x unified",
    )
    return _transparent(fig)


def _alpha_key(text: str) -> str:
    """Clave de orden alfabético español: ignora tildes (Á→a) y mayúsculas SOLO
    para comparar, sin alterar el nombre que se muestra. Así 'Águilas' cae en la A."""
    sin_tilde = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return sin_tilde.casefold()


# ---------------------------------------------------------------------------
inject_style()
payload = load_data()
df_stations = (stations_frame(payload)
               .sort_values("nombre", key=lambda col: col.map(_alpha_key))
               .reset_index(drop=True))
by_code = {s["estacion"]: s for s in payload["estaciones"]}

# Etiqueta del selector: nombre (+ provincia si el nombre se repite, para desambiguar).
dup = df_stations["nombre"].duplicated(keep=False)
label_by_code = {
    r.estacion: (f"{r.nombre} ({r.provincia})" if d else r.nombre)
    for r, d in zip(df_stations.itertuples(index=False), dup)
}
code_by_label = {v: k for k, v in label_by_code.items()}
labels_sorted = sorted(code_by_label, key=_alpha_key)

# --- Selector de idioma (arriba a la derecha) ------------------------------
_, lang_col = st.columns([5, 1])
with lang_col:
    lang_choice = st.segmented_control(
        "Idioma / Language", ["ES", "EN"], default="ES",
        format_func=lambda c: "🇪🇸 ES" if c == "ES" else "🇬🇧 EN",
        label_visibility="collapsed", key="lang",
    )
t = TR["en"] if lang_choice == "EN" else TR["es"]

st.markdown(f'<div class="titlebar"><h1>{t["title"]}</h1></div>', unsafe_allow_html=True)

# --- Bloque 1: selector + mapa (el mapa también filtra al hacer clic) ------
# Leer el clic del mapa del render anterior (el widget guarda su selección en
# st.session_state["map"]) y, si es un clic NUEVO, fijar el valor del selector
# ANTES de crearlo. Así mapa y desplegable quedan sincronizados.
map_state = st.session_state.get("map") or {}
clicked = (map_state.get("selection", {}) or {}).get("points", [])
if clicked:
    p = clicked[0]
    code = (p.get("customdata") or [None])[0]
    if code is None and "point_index" in p:
        code = df_stations.iloc[p["point_index"]]["estacion"]
    if code is not None and code != st.session_state.get("last_map_sel"):
        st.session_state.last_map_sel = code
        st.session_state.sel_label = label_by_code[code]

if "sel_label" not in st.session_state:
    st.session_state.sel_label = labels_sorted[0]

with st.container(border=True, key="card-sel"):
    chosen = st.selectbox(t["select"], labels_sorted, key="sel_label")
selected_code = code_by_label[chosen]

st.plotly_chart(
    map_figure(df_stations, selected_code, t), key="map",
    on_select="rerun", selection_mode="points",
)

# --- Bloque 2: previsión a 7 días ------------------------------------------
s = by_code[selected_code]
fc = forecast_frame(s)

nombre = s["nombre"] or s["estacion"]
alt = f" · {t['elevation']}: {s['altitud']:.0f} m" if s.get("altitud") is not None else ""
st.subheader(f"{t['forecast_for']} {nombre}{alt}")

col_temp, col_prec = st.columns(2, gap="large")
with col_temp:
    with st.container(border=True, key="card-mean"):
        st.markdown(mean_block_html(fc, t), unsafe_allow_html=True)
    with st.container(border=True, key="card-temp"):
        st.markdown(f'<div class="blocktitle">{t["temp_title"]}</div>', unsafe_allow_html=True)
        st.plotly_chart(temp_figure(fc, t), width="stretch")
with col_prec:
    with st.container(border=True, key="card-prec"):
        st.markdown(f'<div class="blocktitle">{t["precip_title"]}</div>', unsafe_allow_html=True)
        st.caption(t["precip_caption"])
        st.plotly_chart(precip_figure(fc, t), width="stretch")

st.markdown(f'<div class="footer">{t["footer"]}</div>', unsafe_allow_html=True)
