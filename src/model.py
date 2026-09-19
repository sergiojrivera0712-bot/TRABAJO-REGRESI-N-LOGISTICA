"""
Paso 3: Modelado - Regresion Lineal.

Entrada: data/processed/dataset.csv (generado por prepare_and_eda.py)

Que hace:
  1. Split train/test reproducible (semilla fija en config.RANDOM_SEED).
  2. Ajusta una Regresion Lineal (sklearn.linear_model.LinearRegression).
  3. Valida los supuestos del modelo sobre el set de entrenamiento:
       - Linealidad       -> grafico residuos vs. valores ajustados
       - Homocedasticidad -> test de Breusch-Pagan (implementacion manual,
                              ya que 'statsmodels' es opcional y no esta
                              disponible en este entorno)
       - Normalidad de residuos -> test de Shapiro-Wilk (scipy) + histograma
       - Multicolinealidad -> Factor de Inflacion de Varianza (VIF), calculado
                              manualmente como 1 / (1 - R^2) de la regresion
                              de cada predictor sobre los demas.
  4. Guarda el modelo ajustado (coeficientes) y el reporte de supuestos.

Salidas:
  - reports/supuestos_modelo.json
  - reports/coeficientes_modelo.csv
  - figures/supuestos_residuos_vs_ajustados.png
  - figures/supuestos_qq_residuos.png
  - data/processed/train.csv, data/processed/test.csv (splits, trazabilidad)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    DATA_PROCESSED_DIR,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    TARGET_COLUMN,
)

plt.rcParams.update({"figure.dpi": 140, "font.size": 9})

TEST_SIZE = 0.2


def breusch_pagan_test(residuals: np.ndarray, X: np.ndarray) -> dict:
    """Implementacion manual del test de Breusch-Pagan (sin statsmodels).

    H0: homocedasticidad (varianza constante de los residuos).
    Se regresan los residuos al cuadrado sobre los predictores y se usa
    el R^2 de esa regresion auxiliar para construir un estadistico
    Chi-cuadrado con k grados de libertad (k = numero de predictores).
    """
    n, k = X.shape
    resid_sq = residuals ** 2
    aux_model = LinearRegression().fit(X, resid_sq)
    r2_aux = aux_model.score(X, resid_sq)
    lm_stat = n * r2_aux
    p_value = 1 - stats.chi2.cdf(lm_stat, df=k)
    return {"estadistico_LM": float(lm_stat), "p_valor": float(p_value), "grados_libertad": int(k)}


def durbin_watson(residuals: np.ndarray) -> float:
    diff = np.diff(residuals)
    return float(np.sum(diff ** 2) / np.sum(residuals ** 2))


def variance_inflation_factors(X: pd.DataFrame) -> dict:
    """VIF manual: para cada predictor, ajusta una regresion sobre los
    demas predictores y calcula 1 / (1 - R^2)."""
    vifs = {}
    for col in X.columns:
        y_aux = X[col].values
        X_aux = X.drop(columns=[col]).values
        r2 = LinearRegression().fit(X_aux, y_aux).score(X_aux, y_aux)
        vifs[col] = float("inf") if r2 >= 1.0 else 1.0 / (1.0 - r2)
    return vifs


def main() -> None:
    df = pd.read_csv(DATA_PROCESSED_DIR / "dataset.csv")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    df.loc[idx_train].to_csv(DATA_PROCESSED_DIR / "train.csv", index=False)
    df.loc[idx_test].to_csv(DATA_PROCESSED_DIR / "test.csv", index=False)

    model = LinearRegression()
    model.fit(X_train, y_train)

    coef_rows = [{"variable": "intercepto", "coeficiente": model.intercept_}]
    for name, coef in zip(FEATURE_COLUMNS, model.coef_):
        coef_rows.append({"variable": name, "coeficiente": coef})
    pd.DataFrame(coef_rows).to_csv(REPORTS_DIR / "coeficientes_modelo.csv", index=False)

    # --- Diagnostico de supuestos sobre el set de entrenamiento ---
    y_train_pred = model.predict(X_train)
    residuals = y_train.values - y_train_pred

    bp = breusch_pagan_test(residuals, X_train.values)
    dw = durbin_watson(residuals)
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    vifs = variance_inflation_factors(X_train)

    supuestos = {
        "homocedasticidad_breusch_pagan": {
            **bp,
            "interpretacion": (
                "No se rechaza H0 (p > 0.05): no hay evidencia fuerte de heterocedasticidad."
                if bp["p_valor"] > 0.05
                else "Se rechaza H0 (p <= 0.05): hay evidencia de heterocedasticidad."
            ),
        },
        "autocorrelacion_durbin_watson": {
            "estadistico": dw,
            "interpretacion": "Cercano a 2 indica ausencia de autocorrelacion de primer orden.",
        },
        "normalidad_residuos_shapiro_wilk": {
            "estadistico": float(shapiro_stat),
            "p_valor": float(shapiro_p),
            "interpretacion": (
                "No se rechaza H0 (p > 0.05): los residuos son consistentes con una distribucion normal."
                if shapiro_p > 0.05
                else "Se rechaza H0 (p <= 0.05): los residuos se desvian de la normalidad."
            ),
        },
        "multicolinealidad_vif": {
            **{k: (v if v != float("inf") else "inf") for k, v in vifs.items()},
            "interpretacion": "VIF > 10 sugiere multicolinealidad problematica; VIF > 5 amerita atencion.",
        },
    }
    with open(REPORTS_DIR / "supuestos_modelo.json", "w", encoding="utf-8") as f:
        json.dump(supuestos, f, ensure_ascii=False, indent=2)
    print("Supuestos del modelo:", json.dumps(supuestos, ensure_ascii=False, indent=2))

    # --- Graficos de diagnostico ---
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(y_train_pred, residuals, s=16, alpha=0.6, color="#1C7293")
    ax.axhline(0, color="crimson", linestyle="--", linewidth=1)
    ax.set_xlabel("Valores ajustados (entrenamiento)")
    ax.set_ylabel("Residuos")
    ax.set_title("Residuos vs. valores ajustados (linealidad / homocedasticidad)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "supuestos_residuos_vs_ajustados.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("QQ-plot de residuos (normalidad)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "supuestos_qq_residuos.png")
    plt.close(fig)

    print(f"\nModelo ajustado con {len(X_train)} obs. de entrenamiento y {len(X_test)} de prueba.")
    print("Coeficientes guardados en reports/coeficientes_modelo.csv")


if __name__ == "__main__":
    main()
