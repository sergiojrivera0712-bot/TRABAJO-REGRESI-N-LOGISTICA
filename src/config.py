"""
Configuracion central del proyecto: semilla de aleatoriedad, año de analisis,
indicadores del Banco Mundial y rutas relativas.

Mantener todas las "constantes mágicas" en un solo lugar facilita la
reproducibilidad y la trazabilidad exigidas en el informe.
"""

from pathlib import Path

# ------------------------------------------------------------------
# Reproducibilidad
# ------------------------------------------------------------------
RANDOM_SEED = 42

# ------------------------------------------------------------------
# Rutas relativas (nunca absolutas) — todo cuelga de la raiz del repo
# ------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FIGURES_DIR = ROOT_DIR / "figures"
REPORTS_DIR = ROOT_DIR / "reports"

for _dir in (DATA_RAW_DIR, DATA_PROCESSED_DIR, FIGURES_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Fuente de datos: API REST del Banco Mundial (World Bank Open Data)
# https://api.worldbank.org/v2/  — no requiere autenticacion ni API key.
# ------------------------------------------------------------------
API_BASE_URL = "https://api.worldbank.org/v2"
YEAR = 2019  # ultimo año pre-pandemia con buena cobertura para todos los indicadores

# Variable objetivo (continua)
TARGET = {
    "code": "SP.DYN.LE00.IN",
    "name": "esperanza_vida",
    "label": "Esperanza de vida al nacer (años)",
}

# Predictores (3-6 variables continuas)
PREDICTORS = [
    {"code": "NY.GDP.PCAP.CD", "name": "pib_per_capita", "label": "PIB per capita (US$ corrientes)"},
    {"code": "SH.XPD.CHEX.PC.CD", "name": "gasto_salud_per_capita", "label": "Gasto en salud per capita (US$ corrientes)"},
    {"code": "SP.URB.TOTL.IN.ZS", "name": "poblacion_urbana_pct", "label": "Poblacion urbana (% del total)"},
    {"code": "SE.SEC.ENRR", "name": "matricula_secundaria_pct", "label": "Matricula bruta en secundaria (%)"},
    {"code": "IT.NET.USER.ZS", "name": "usuarios_internet_pct", "label": "Usuarios de internet (% de la poblacion)"},
]

ALL_INDICATORS = [TARGET] + PREDICTORS

FEATURE_COLUMNS = [p["name"] for p in PREDICTORS]
TARGET_COLUMN = TARGET["name"]
