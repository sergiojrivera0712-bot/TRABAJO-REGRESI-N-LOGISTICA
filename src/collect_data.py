"""
Paso 1: Recoleccion de datos.

Fuente: API REST publica del Banco Mundial (World Bank Open Data),
https://api.worldbank.org/v2/ . No requiere API key.

Para cada indicador se descarga el valor mas reciente disponible para el
anio definido en config.YEAR, para TODOS los paises (parametro country=all).
El listado de paises se descarga por separado (endpoint /country) y se usa
para filtrar las filas que corresponden a agregados regionales (p. ej.
"World", "OECD members", "Latin America & Caribbean") en lugar de paises
reales — estos agregados se identifican porque su campo region.value es
"Aggregates".

Criterios de calidad de la recoleccion (trazabilidad):
  1. Se registra la URL exacta, la fecha/hora de descarga y el codigo de
     respuesta HTTP de cada llamada (ver data/raw/collection_log.csv).
  2. Se guarda el JSON crudo de cada respuesta en data/raw/ sin modificar,
     para permitir auditar el dato tal como lo entrego la fuente.
  3. Solo se conservan como "pais" los registros cuyo region.value no sea
     "Aggregates" (evita mezclar paises con bloques regionales).
  4. Se usa una unica ejecucion con fecha fija (config.YEAR) para que el
     dataset sea reproducible entre ejecuciones distintas del script.

Salida: data/raw/<indicador>_raw.json (uno por indicador) y
        data/raw/countries_raw.json (metadatos de pais)
"""

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


sys.path.append(str(Path(__file__).resolve().parent))
from config import ALL_INDICATORS, API_BASE_URL, DATA_RAW_DIR, YEAR  # noqa: E402

TIMEOUT_SECONDS = 30
MAX_RETRIES = 3


def _log_row(log_path: Path, url: str, status: int, note: str = "") -> None:
    is_new = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp_utc", "url", "http_status", "nota"])
        writer.writerow([datetime.now(timezone.utc).isoformat(), url, status, note])


def _get_with_retries(url: str, log_path: Path) -> dict:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=TIMEOUT_SECONDS)
            _log_row(log_path, url, resp.status_code, f"intento {attempt}")
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:  # pragma: no cover
            last_exc = exc
            _log_row(log_path, url, -1, f"error intento {attempt}: {exc}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"No se pudo descargar {url} tras {MAX_RETRIES} intentos") from last_exc


def fetch_indicator(indicator_code: str, year: int, log_path: Path) -> list:
    """Descarga un indicador para todos los paises en un anio dado."""
    url = f"{API_BASE_URL}/country/all/indicator/{indicator_code}?format=json&date={year}&per_page=400"
    payload = _get_with_retries(url, log_path)
    # payload = [metadata, records]
    return payload[1] if len(payload) > 1 and payload[1] else []


def fetch_country_metadata(log_path: Path) -> list:
    """Descarga el listado completo de paises/agregados (paginado)."""
    records = []
    page = 1
    while True:
        url = f"{API_BASE_URL}/country?format=json&per_page=200&page={page}"
        payload = _get_with_retries(url, log_path)
        meta, page_records = payload[0], payload[1]
        records.extend(page_records)
        if page >= meta["pages"]:
            break
        page += 1
    return records


def main() -> None:
    log_path = DATA_RAW_DIR / "collection_log.csv"

    print(f"Descargando metadatos de paises desde {API_BASE_URL}/country ...")
    countries = fetch_country_metadata(log_path)
    with open(DATA_RAW_DIR / "countries_raw.json", "w", encoding="utf-8") as f:
        json.dump(countries, f, ensure_ascii=False, indent=2)
    print(f"  -> {len(countries)} registros de pais/agregado guardados.")

    for indicator in ALL_INDICATORS:
        code, name = indicator["code"], indicator["name"]
        print(f"Descargando indicador {code} ({name}) para el anio {YEAR} ...")
        records = fetch_indicator(code, YEAR, log_path)
        out_path = DATA_RAW_DIR / f"{name}_raw.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"  -> {len(records)} registros guardados en {out_path.relative_to(DATA_RAW_DIR.parents[1])}")

    print("\nRecoleccion completa. Revisa data/raw/collection_log.csv para la trazabilidad de cada llamada HTTP.")


if __name__ == "__main__":
    main()
