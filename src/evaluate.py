"""
Paso 4: Evaluacion del modelo sobre el set de prueba (holdout).

Entrada: data/processed/train.csv y data/processed/test.csv (generados por
         model.py, con la misma semilla de aleatoriedad).

Metricas: MAE, MSE, RMSE, R^2 (sobre train y sobre test, para detectar
          sobreajuste).

Graficos requeridos:
  - Dispersion de valores reales vs. predichos + linea de referencia y=x
    (equivalente a "dispersion + recta" para un modelo multivariable).
  - Residuos vs. predicciones (set de prueba).

Salidas:
  - reports/metricas_evaluacion.json
  - figures/evaluacion_real_vs_predicho.png
  - figures/evaluacion_residuos_vs_predicciones.png
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(str(Path(__file__).resolve().parent))
from config import DATA_PROCESSED_DIR, FEATURE_COLUMNS, FIGURES_DIR, REPORTS_DIR, TARGET_COLUMN  # noqa: E402

plt.rcParams.update({"figure.dpi": 140, "font.size": 9})


def compute_metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "MSE": float(mse), "RMSE": rmse, "R2": float(r2)}


def main() -> None:
    train = pd.read_csv(DATA_PROCESSED_DIR / "train.csv")
    test = pd.read_csv(DATA_PROCESSED_DIR / "test.csv")

    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        "entrenamiento": compute_metrics(y_train, y_train_pred),
        "prueba": compute_metrics(y_test, y_test_pred),
        "n_entrenamiento": int(len(y_train)),
        "n_prueba": int(len(y_test)),
    }
    with open(REPORTS_DIR / "metricas_evaluacion.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("Metricas de evaluacion:", json.dumps(metrics, ensure_ascii=False, indent=2))

    # --- Grafico 1: valores reales vs. predichos (set de prueba) ---
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_test, y_test_pred, s=22, alpha=0.7, color="#1C7293", label="paises (prueba)")
    lims = [min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())]
    ax.plot(lims, lims, color="crimson", linestyle="--", linewidth=1.3, label="y = x (prediccion perfecta)")
    ax.set_xlabel("Esperanza de vida real (anios)")
    ax.set_ylabel("Esperanza de vida predicha (anios)")
    ax.set_title(f"Real vs. predicho (set de prueba, n={len(y_test)})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "evaluacion_real_vs_predicho.png")
    plt.close(fig)

    # --- Grafico 2: residuos vs. predicciones (set de prueba) ---
    residuals_test = y_test.values - y_test_pred
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(y_test_pred, residuals_test, s=22, alpha=0.7, color="#2C5F2D")
    ax.axhline(0, color="crimson", linestyle="--", linewidth=1.3)
    ax.set_xlabel("Valores predichos (set de prueba)")
    ax.set_ylabel("Residuos (real - predicho)")
    ax.set_title("Residuos vs. predicciones (set de prueba)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "evaluacion_residuos_vs_predicciones.png")
    plt.close(fig)

    print("\nGraficos de evaluacion guardados en figures/.")


if __name__ == "__main__":
    main()
