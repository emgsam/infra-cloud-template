"""API de inferencia de ventas perdidas.

Endpoints:
  POST /predict                  qué accesorios probablemente le faltaron a una compra
  GET  /healthz                  200 con el modelo cargado, 503 mientras carga
  GET  /metrics                  formato Prometheus
  GET  /versions                 versiones guardadas, con sus métricas y cuál está activa
  POST /versions/{v}/activate    activa una versión y la recarga en caliente

El modelo NO está en la imagen: al arrancar se lee <MODELS_URI>/active.json y se
carga esa versión desde Cloud Storage (o desde una carpeta, en local). Cada
RELOAD_SECONDS se vuelve a mirar active.json, así todas las réplicas terminan
sirviendo la misma versión después de un /activate.
"""

from __future__ import annotations

import io
import json
import os
import socket
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field, field_validator

from ventas import almacen, registro
from ventas.modelo import sugerir

PORT = int(os.environ.get("PORT", "8000"))
POD = os.environ.get("POD_NAME", socket.gethostname())

REQUESTS = Counter("ventas_requests_total", "Peticiones por endpoint y código", ["endpoint", "status"])
LATENCIA = Histogram("ventas_request_seconds", "Latencia por endpoint", ["endpoint"],
                     buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10))
SUGERENCIAS = Counter("ventas_sugerencias_total", "Sugerencias devueltas por prioridad", ["prioridad"])
VERSION_ACTIVA = Gauge("ventas_modelo_info", "Versión cargada (valor 1)", ["version"])
DESCARTES = Counter("ventas_db_descartes_total", "Filas que no se pudieron registrar en la base")

escritor = registro.EscritorAsincrono(al_descartar=DESCARTES.inc)
_estado: dict[str, Any] = {"bundle": None, "version": None}
_lock = threading.Lock()


# ---------------------------------------------------------------- carga del modelo
def _modelos() -> str:
    return os.environ.get("MODELS_URI", "../modelos")


def _cada() -> float:
    return float(os.environ.get("RELOAD_SECONDS", "30"))


def _version_en_puntero() -> str | None:
    ruta = almacen.unir(_modelos(), "active.json")
    return json.loads(almacen.leer(ruta))["version"] if almacen.existe(ruta) else None


def cargar(version: str) -> None:
    bundle = joblib.load(io.BytesIO(almacen.leer(almacen.unir(_modelos(), version, "model.joblib"))))
    for modelo in bundle["modelos"].values():
        modelo.n_jobs = 1  # una fila por petición: paralelizar entre árboles solo agrega overhead
    with _lock:
        anterior = _estado["version"]
        _estado.update(bundle=bundle, version=version)
    if anterior:
        VERSION_ACTIVA.remove(anterior)
    VERSION_ACTIVA.labels(version).set(1)
    print(f"modelo {version} cargado ({len(bundle['confiables'])} accesorios confiables)", flush=True)


def _vigilar_puntero() -> None:
    while True:
        time.sleep(_cada())
        try:
            v = _version_en_puntero()
            if v and v != _estado["version"]:
                cargar(v)
        except Exception as exc:
            print(f"no se pudo recargar el modelo: {exc}", flush=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        v = _version_en_puntero()
        if v:
            cargar(v)
        else:
            print(f"no hay active.json en {_modelos()}: correr el trainer primero", flush=True)
    except Exception as exc:  # /healthz queda en 503 y el motivo en el log
        print(f"modelo no cargado: {exc}", flush=True)
    escritor.iniciar()
    if _cada() > 0:
        threading.Thread(target=_vigilar_puntero, daemon=True, name="reload").start()
    yield


app = FastAPI(title="ventas-perdidas", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def medir(request: Request, call_next):
    inicio = time.perf_counter()
    respuesta = await call_next(request)
    ruta = request.scope.get("route")
    endpoint = ruta.path if ruta else "otro"
    if endpoint != "/metrics":
        REQUESTS.labels(endpoint, str(respuesta.status_code)).inc()
        LATENCIA.labels(endpoint).observe(time.perf_counter() - inicio)
    return respuesta


# ------------------------------------------------------------------------ esquemas
class Articulo(BaseModel):
    codigo: str = Field(min_length=1, examples=["DUR-319093"])
    cantidad: float = Field(default=1, gt=0)


class PedidoPrediccion(BaseModel):
    cliente: int | None = Field(default=None, examples=[1001])
    mes: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}$", examples=["2026-09"])
    articulos: list[Articulo] = Field(default_factory=list)
    productos: list[str] = Field(default_factory=list,
                                 description="productos base, como alternativa a los códigos")

    @field_validator("productos")
    @classmethod
    def _sin_vacios(cls, v: list[str]) -> list[str]:
        return [p.strip() for p in v if p.strip()]


# ----------------------------------------------------------------------- endpoints
@app.get("/")
def index() -> dict[str, Any]:
    return {"servicio": "ventas-perdidas", "version_activa": _estado["version"],
            "endpoints": ["POST /predict", "GET /healthz", "GET /metrics", "GET /versions",
                          "POST /versions/{v}/activate", "GET /docs"]}


@app.get("/healthz")
def healthz() -> Response:
    if _estado["bundle"] is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return JSONResponse({"status": "ok", "version": _estado["version"], "pod": POD})


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(pedido: PedidoPrediccion) -> dict[str, Any]:
    inicio = time.perf_counter()
    with _lock:
        bundle, version = _estado["bundle"], _estado["version"]
    if bundle is None:
        raise HTTPException(503, "modelo no cargado todavía; ver /healthz y los logs")
    if not pedido.articulos and not pedido.productos:
        raise HTTPException(422, "mandá al menos un artículo (articulos) o un producto base (productos)")

    mapa = bundle["mapa_articulos"]
    reconocidos = {mapa[a.codigo] for a in pedido.articulos if a.codigo in mapa}
    desconocidos = sorted({a.codigo for a in pedido.articulos if a.codigo not in mapa})
    columnas = set(bundle["columnas"])
    reconocidos |= {p for p in pedido.productos if p in columnas}
    desconocidos += sorted({p for p in pedido.productos if p not in columnas})

    sugerencias = sugerir(bundle, reconocidos) if reconocidos else []
    for s in sugerencias:
        SUGERENCIAS.labels(s["prioridad"]).inc()
    latencia_ms = round((time.perf_counter() - inicio) * 1000, 2)

    escritor.encolar({"ts": datetime.now(timezone.utc), "endpoint": "/predict", "status": 200,
                      "latencia_ms": latencia_ms, "version": version, "pod": POD,
                      "cliente": pedido.cliente, "mes": pedido.mes,
                      "n_articulos": len(pedido.articulos) + len(pedido.productos)}, sugerencias)
    return {"version": version, "cliente": pedido.cliente, "mes": pedido.mes,
            "productos_reconocidos": sorted(reconocidos), "no_reconocidos": desconocidos,
            "sugerencias": sugerencias, "latencia_ms": latencia_ms}


@app.get("/versions")
def versions() -> dict[str, Any]:
    activa = _version_en_puntero()
    salida = []
    for v in almacen.listar_versiones(_modelos()):
        meta = json.loads(almacen.leer(almacen.unir(_modelos(), v, "metadata.json")))
        m = meta.get("metricas", {})
        salida.append({"version": v, "mes_corte": meta.get("mes_corte"),
                       "f1_test": m.get("f1_test"), "f1_baseline_test": m.get("f1_baseline_test"),
                       "resultado": meta.get("resultado"), "creada": meta.get("fin"),
                       "activa": v == activa})
    return {"activa": activa, "cargada_en_este_pod": _estado["version"], "versiones": salida}


@app.post("/versions/{version}/activate")
def activate(version: str) -> dict[str, Any]:
    if version not in almacen.listar_versiones(_modelos()):
        raise HTTPException(404, f"no existe la versión {version}")
    cargar(version)  # primero carga: si el archivo está roto, el puntero no cambia
    almacen.escribir(almacen.unir(_modelos(), "active.json"),
                     json.dumps({"version": version, "activada": datetime.now(timezone.utc)
                                 .isoformat(timespec="seconds"), "manual": True}).encode())
    registro.marcar_activa(version)
    return {"activa": version, "nota": f"las otras réplicas la cargan en menos de {_cada():.0f} s"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
