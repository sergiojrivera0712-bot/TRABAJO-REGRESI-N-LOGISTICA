# Regresion Lineal: Esperanza de vida al nacer segun indicadores socioeconomicos

Proyecto academico de ciencia de datos que documenta el flujo completo de un
modelo de **Regresion Lineal**, desde la obtencion de datos hasta la
evaluacion, usando datos abiertos del **Banco Mundial (World Bank Open
Data)**.

- **Variable objetivo (continua):** esperanza de vida al nacer (años), anio 2019.
- **Predictores (5):** PIB per capita, gasto en salud per capita, poblacion
  urbana (%), matricula bruta en secundaria (%), usuarios de internet (%).
- **Unidad de analisis:** pais (n = 131 paises tras la limpieza).

El informe completo en PDF esta en `Informe_RegresionLineal_PasosPazos.pdf`
(raiz del repositorio).

## Estructura del proyecto

```
regresion_proyecto/
├── README.md
├── requirements.txt
├── Informe_RegresionLineal_PasosPazos.pdf
├── src/
│   ├── config.py            # constantes, semilla, rutas relativas, indicadores
│   ├── collect_data.py      # Paso 1: recoleccion de datos (API Banco Mundial)
│   ├── prepare_and_eda.py   # Paso 2: limpieza, diccionario de variables, EDA
│   ├── model.py             # Paso 3: split train/test, ajuste, supuestos
│   └── evaluate.py          # Paso 4: metricas (MAE, MSE, RMSE, R2) y graficos
├── data/
│   ├── raw/                 # JSON crudos descargados de la API + log de llamadas
│   └── processed/           # dataset.csv, train.csv, test.csv, diccionario
├── figures/                 # graficos de EDA, supuestos y evaluacion (.png)
└── reports/                 # metricas y estadisticos en .json / .csv
```

## Fuente de datos y trazabilidad

Los datos provienen de la API REST publica del Banco Mundial
(`https://api.worldbank.org/v2/`), que no requiere autenticacion. Se
descargan los metadatos de pais (para excluir agregados regionales como
"World" u "OECD members") y los 5+1 indicadores definidos en `config.py`.

Cada llamada HTTP realizada por `collect_data.py` queda registrada en
`data/raw/collection_log.csv` (marca de tiempo UTC, URL exacta, codigo de
respuesta HTTP). Los JSON crudos de la respuesta se guardan sin modificar en
`data/raw/` para permitir auditar el dato tal como lo entrego la fuente.

Criterios de calidad aplicados en `prepare_and_eda.py` (ver tambien
`reports/quality_log.json`):

1. Se conservan solo paises reales (se excluyen agregados regionales/de
   ingreso segun el campo `region.value` de la API).
2. Se eliminan duplicados exactos por pais.
3. Se descartan filas con datos faltantes en la variable objetivo o en
   cualquiera de los 5 predictores.
4. Se descartan valores no fisicamente posibles (defensivo): esperanza de
   vida fuera de (0, 120) años, predictores negativos.

Resultado: de 260 registros originales del indicador objetivo, quedan
**131 paises** con datos completos y validos en las 6 variables.

## Como ejecutar

Requiere Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Paso 1: recolectar datos desde la API del Banco Mundial
python src/collect_data.py

# Paso 2: limpiar datos y generar el EDA (graficos + estadisticos)
python src/prepare_and_eda.py

# Paso 3: dividir train/test, ajustar el modelo y validar supuestos
python src/model.py

# Paso 4: evaluar el modelo (metricas + graficos) sobre el set de prueba
python src/evaluate.py
```

Todos los scripts usan **rutas relativas** (definidas en `src/config.py` a
partir de la ubicacion del propio archivo) y una **semilla de aleatoriedad
fija** (`RANDOM_SEED = 42`), por lo que el resultado es reproducible entre
ejecuciones.

## Resultados principales (resumen)

| Metrica | Entrenamiento | Prueba |
|---|---|---|
| MAE  | 2.49 | 2.31 |
| MSE  | 11.65 | 8.64 |
| RMSE | 3.41 | 2.94 |
| R²   | 0.71 | 0.82 |

Ver el informe en PDF para el detalle completo de metodologia, supuestos,
graficos y conclusiones.

## Licencia y uso

Datos: World Bank Open Data (licencia CC BY-4.0). Codigo: uso academico.
