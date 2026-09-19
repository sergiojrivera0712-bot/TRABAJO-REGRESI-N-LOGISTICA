"""
Paso 2: Preparacion y analisis exploratorio de datos (EDA).

Entradas: los JSON crudos en data/raw/ producidos por collect_data.py
  - countries_raw.json: metadatos de pais (para filtrar agregados regionales)
  - <indicador>_raw.json: un archivo por variable (objetivo + predictores)

Salidas:
  - data/processed/dataset.csv        -> dataset limpio listo para modelar
  - data/processed/diccionario_variables.csv
  - figures/eda_*.png                 -> graficos exploratorios
  - reports/eda_summary.json          -> estadisticos descriptivos y correlaciones
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    ALL_INDICATORS,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    REPORTS_DIR,
    TARGET_COLUMN,
)

plt.rcParams.update({"figure.dpi": 140, "font.size": 9})


def load_indicator_records(name: str) -> pd.DataFrame:
    path = DATA_RAW_DIR / f"{name}_raw.json"
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    rows = []
    for r in records:
        rows.append(
            {
                "iso3": r.get("countryiso3code"),
                "pais": r["country"]["value"],
                name: r["value"],
            }
        )
    return pd.DataFrame(rows)


def load_real_country_codes() -> set:
    """Codigos ISO3 que corresponden a PAISES reales (no agregados regionales)."""
    with open(DATA_RAW_DIR / "countries_raw.json", encoding="utf-8") as f:
        countries = json.load(f)
    return {
        c["id"]
        for c in countries
        if c.get("region", {}).get("value", "Aggregates") != "Aggregates"
    }


def build_dataset() -> pd.DataFrame:
    real_codes = load_real_country_codes()

    merged = None
    n_raw_target = None
    for indicator in ALL_INDICATORS:
        df = load_indicator_records(indicator["name"])
        # Descarta filas sin codigo iso3 antes de unir: varios agregados del
        # Banco Mundial (p. ej. algunas entradas de "Aggregates") no traen
        # countryiso3code y produce combinaciones espurias en el merge.
        df = df[df["iso3"].notna() & (df["iso3"] != "")]
        if indicator["name"] == TARGET_COLUMN:
            n_raw_target = len(df)
        merged = df if merged is None else merged.merge(df[["iso3", indicator["name"]]], on="iso3", how="outer")

    # Criterio de calidad 1: solo paises reales (excluye agregados regionales
    # como "World", "OECD members", "Latin America & Caribbean", etc.)
    before = len(merged)
    merged = merged[merged["iso3"].isin(real_codes)].copy()
    dropped_aggregates = before - len(merged)

    # Criterio de calidad 2: eliminar duplicados exactos por pais
    merged = merged.drop_duplicates(subset=["iso3"])

    # Criterio de calidad 3: eliminar filas con valores faltantes en
    # cualquiera de las variables del modelo (objetivo o predictores)
    model_cols = [TARGET_COLUMN] + FEATURE_COLUMNS
    before_na = len(merged)
    merged = merged.dropna(subset=model_cols).copy()
    dropped_na = before_na - len(merged)

    # Criterio de calidad 4: descartar valores no fisicos (defensivo)
    merged = merged[(merged[TARGET_COLUMN] > 0) & (merged[TARGET_COLUMN] < 120)]
    merged = merged[merged[FEATURE_COLUMNS].ge(0).all(axis=1)]

    merged = merged.reset_index(drop=True)

    quality_log = {
        "registros_indicador_objetivo_banco_mundial": int(n_raw_target),
        "registros_tras_union_de_las_6_variables": int(before),
        "descartados_por_ser_agregado_regional": int(dropped_aggregates),
        "descartados_por_datos_faltantes": int(dropped_na),
        "paises_finales_en_dataset": int(len(merged)),
    }
    with open(REPORTS_DIR / "quality_log.json", "w", encoding="utf-8") as f:
        json.dump(quality_log, f, ensure_ascii=False, indent=2)
    print("Log de calidad de datos:", quality_log)

    return merged


def write_data_dictionary() -> None:
    rows = [
        {
            "variable": "pais",
            "tipo": "texto",
            "descripcion": "Nombre del pais",
            "fuente": "World Bank API /country",
        },
        {
            "variable": "iso3",
            "tipo": "texto",
            "descripcion": "Codigo ISO 3166-1 alfa-3 del pais",
            "fuente": "World Bank API /country",
        },
    ]
    for ind in ALL_INDICATORS:
        rows.append(
            {
                "variable": ind["name"],
                "tipo": "numerica continua",
                "descripcion": ind["label"],
                "fuente": f"World Bank API, indicador {ind['code']}",
            }
        )
    pd.DataFrame(rows).to_csv(DATA_PROCESSED_DIR / "diccionario_variables.csv", index=False)


def run_eda(df: pd.DataFrame) -> None:
    model_cols = [TARGET_COLUMN] + FEATURE_COLUMNS

    desc = df[model_cols].describe().T
    desc.to_csv(REPORTS_DIR / "estadisticos_descriptivos.csv")

    corr = df[model_cols].corr(numeric_only=True)
    corr.to_csv(REPORTS_DIR / "matriz_correlacion.csv")

    # Heatmap de correlaciones
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(model_cols)))
    ax.set_yticks(range(len(model_cols)))
    ax.set_xticklabels(model_cols, rotation=45, ha="right")
    ax.set_yticklabels(model_cols)
    for i in range(len(model_cols)):
        for j in range(len(model_cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                     color="white" if abs(corr.values[i, j]) > 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlacion de Pearson")
    ax.set_title("Matriz de correlacion — variables del modelo")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_correlacion.png")
    plt.close(fig)

    # Histograma de la variable objetivo
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(df[TARGET_COLUMN], bins=20, color="#2C5F2D", edgecolor="white")
    ax.set_xlabel("Esperanza de vida al nacer (anios)")
    ax.set_ylabel("numero de paises")
    ax.set_title("Distribucion de la variable objetivo")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_hist_objetivo.png")
    plt.close(fig)

    # Dispersion de cada predictor vs la variable objetivo
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    for ax, col in zip(axes.flat, FEATURE_COLUMNS):
        ax.scatter(df[col], df[TARGET_COLUMN], s=14, alpha=0.6, color="#1C7293")
        ax.set_xlabel(col)
        ax.set_ylabel(TARGET_COLUMN)
    for ax in axes.flat[len(FEATURE_COLUMNS):]:
        ax.axis("off")
    fig.suptitle("Dispersion: predictores vs. esperanza de vida")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_dispersion_predictores.png")
    plt.close(fig)

    print("EDA generada en figures/ y reports/.")


def main() -> None:
    df = build_dataset()
    write_data_dictionary()
    df.to_csv(DATA_PROCESSED_DIR / "dataset.csv", index=False)
    print(f"Dataset final: {df.shape[0]} paises x {df.shape[1]} columnas -> data/processed/dataset.csv")
    run_eda(df)


if __name__ == "__main__":
    main()
