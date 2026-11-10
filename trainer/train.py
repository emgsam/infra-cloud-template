"""Servicio de entrenamiento: entrenar -> evaluar -> versionar -> guardar -> registrar.

Corre como Job de Kubernetes (o con Docker Compose en local). Cada corrida:
  1. carga el extracto de facturas y arma la tabla de compras cliente-mes
  2. entrena un Random Forest por accesorio con los meses hasta --mes-corte
  3. evalúa en train y en el/los mes(es) siguientes (test), contra la regresión
     logística, la línea base y la versión activa (re-evaluada sobre el mismo test)
  4. guarda model.joblib + metadata.json en <modelos>/vAAAAMMDD_HHMMSS/
  5. si supera a la línea base y no empeora a la activa, actualiza <modelos>/active.json

Ejemplos:
  python train.py --datos ../datos --modelos ../modelos --mes-corte 2026-08
  python train.py --datos gs://BUCKET/data --modelos gs://BUCKET/models --mes-corte 2026-08
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version as version_lib

import joblib

from ventas import almacen
from ventas.datos import armar_compras, cargar_lineas, separar
from ventas.modelo import decidir, entrenar, evaluar_bundle

LIBRERIAS = ["scikit-learn", "pandas", "numpy", "joblib"]
MESES_VALIDACION = 3


def _args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datos", default=os.environ.get("DATA_URI", "../datos"),
                   help="carpeta o gs:// con Export_seco_anonimizado.xlsx y productos.xlsx")
    p.add_argument("--modelos", default=os.environ.get("MODELS_URI", "../modelos"),
                   help="carpeta o gs:// donde se guardan las versiones")
    p.add_argument("--mes-corte", default=os.environ.get("MES_CORTE"), required=not os.environ.get("MES_CORTE"),
                   help="último mes de entrenamiento, AAAA-MM")
    p.add_argument("--meses-test", type=int, default=int(os.environ.get("MESES_TEST", "1")),
                   help="meses siguientes al corte que se usan como test (default 1)")
    p.add_argument("--sin-busqueda", action="store_true",
                   default=os.environ.get("SIN_BUSQUEDA", "") == "1",
                   help="usa los hiperparámetros globales del Curso 1 en vez de buscar por accesorio")
    p.add_argument("--version", default=None, help="nombre de la versión (default: fecha y hora UTC)")
    return p.parse_args(argv)


def _version_activa(modelos: str) -> str | None:
    ruta = almacen.unir(modelos, "active.json")
    if not almacen.existe(ruta):
        return None
    return json.loads(almacen.leer(ruta))["version"]


def main(argv=None) -> dict:
    a = _args(argv)
    inicio = datetime.now(timezone.utc)
    version = a.version or inicio.strftime("v%Y%m%d_%H%M%S")
    print(f"== {version}: corte {a.mes_corte}, test {a.meses_test} mes(es), datos {a.datos}", flush=True)

    # 1. Cargar y preparar
    lineas, productos = cargar_lineas(a.datos)
    compras = armar_compras(lineas, productos)
    train, test, meses_test = separar(compras, a.mes_corte, a.meses_test)
    print(f"compras: train {len(train)} | test {len(test)} ({', '.join(meses_test)})", flush=True)

    validacion = None
    if not a.sin_busqueda:
        meses_train = compras.info.loc[train.index, "mes"]
        ultimos = sorted(meses_train.unique())[-MESES_VALIDACION:]
        es_valid = meses_train.isin(ultimos)
        validacion = (train[~es_valid], train[es_valid])
        print(f"búsqueda por accesorio; validación con {[str(m) for m in ultimos]}", flush=True)

    # 2 y 3. Entrenar y evaluar
    salida = entrenar(train, test, compras.accesorios, validacion)
    bundle, resultado = salida["bundle"], salida["resultado"]
    bundle["version"] = version
    bundle["mapa_articulos"] = compras.mapa_articulos
    m = resultado["metricas"]

    # Comparación con la versión activa, sobre el mismo período de test
    anterior = _version_activa(a.modelos)
    f1_anterior = None
    if anterior:
        previo = joblib.load(io.BytesIO(almacen.leer(almacen.unir(a.modelos, anterior, "model.joblib"))))
        f1_anterior = round(evaluar_bundle(previo, test), 4)
    decision, motivo = decidir(m["f1_test"], m["f1_baseline_test"], f1_anterior)

    # 4. Versionar y guardar
    metadata = {
        "version": version,
        "tipo": "sklearn-random-forest-por-accesorio",
        "mes_corte": a.mes_corte,
        "meses_test": meses_test,
        "compras_train": len(train),
        "compras_test": len(test),
        **resultado,
        "version_anterior": anterior,
        "f1_version_anterior_test": f1_anterior,
        "resultado": decision,
        "motivo": motivo,
        "inicio": inicio.isoformat(timespec="seconds"),
        "fin": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "librerias": {lib: version_lib(lib) for lib in LIBRERIAS},
        "git_sha": os.environ.get("GIT_SHA", "local"),
    }
    buffer = io.BytesIO()
    joblib.dump(bundle, buffer, compress=3)
    almacen.escribir(almacen.unir(a.modelos, version, "model.joblib"), buffer.getvalue())
    almacen.escribir(almacen.unir(a.modelos, version, "metadata.json"),
                     json.dumps(metadata, ensure_ascii=False, indent=2).encode())

    # 5. Registrar: el puntero active.json solo cambia si la versión se activa
    if decision == "activada":
        almacen.escribir(almacen.unir(a.modelos, "active.json"),
                         json.dumps({"version": version, "activada": metadata["fin"]}).encode())

    print(f"F1 test {m['f1_test']:.3f} | train {m['f1_train']:.3f} | logística {m['f1_logistica_test']:.3f}"
          f" | base {m['f1_baseline_test']:.3f} | anterior {f1_anterior}", flush=True)
    print(f"confiables (F1 >= 0,5): {len(resultado['accesorios_confiables'])} de {len(compras.accesorios)}")
    print(f"duración {resultado['duracion_entrenamiento_s']} s -> {decision}: {motivo}", flush=True)
    return metadata


if __name__ == "__main__":
    main(sys.argv[1:])
