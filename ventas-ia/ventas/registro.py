"""Registro en PostgreSQL (sección 2.4 de la consigna).

Cuatro tablas: requests, predictions, model_versions y trainings. Si no hay
DATABASE_URL (tests, o correr sin base) todo es un no-op, así el servicio sigue
funcionando.

La API escribe de forma ASÍNCRONA: /predict deja la fila en una cola y un hilo
aparte la inserta. Así la latencia que miden las pruebas de carga es la de la
API y no la de Postgres. Si la cola se llena (la base no da abasto), se descartan
filas y se cuenta en una métrica, en lugar de frenar las respuestas.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from typing import Any

ESQUEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id          BIGSERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
    endpoint    TEXT NOT NULL,
    status      INT NOT NULL,
    latencia_ms DOUBLE PRECISION NOT NULL,
    version     TEXT,
    pod         TEXT,
    cliente     BIGINT,
    mes         TEXT,
    n_articulos INT
);
CREATE TABLE IF NOT EXISTS predictions (
    id          BIGSERIAL PRIMARY KEY,
    request_id  BIGINT REFERENCES requests(id),
    cliente     BIGINT,
    accesorio   TEXT NOT NULL,
    prob        DOUBLE PRECISION NOT NULL,
    prioridad   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_versions (
    version     TEXT PRIMARY KEY,
    gcs_uri     TEXT NOT NULL,
    f1_train    DOUBLE PRECISION,
    f1_test     DOUBLE PRECISION,
    f1_baseline DOUBLE PRECISION,
    activa      BOOLEAN NOT NULL DEFAULT false,
    creada      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS trainings (
    id              BIGSERIAL PRIMARY KEY,
    version         TEXT NOT NULL,
    inicio          TIMESTAMPTZ,
    fin             TIMESTAMPTZ,
    duracion_s      DOUBLE PRECISION,
    hiperparametros JSONB,
    mes_corte       TEXT,
    resultado       TEXT,
    motivo          TEXT,
    metadata        JSONB
);
CREATE INDEX IF NOT EXISTS requests_ts ON requests (ts);
"""


def _conectar():
    url = os.environ.get("DATABASE_URL")
    if not url:
        return None
    import psycopg  # import diferido: el trainer sin base no lo necesita

    conn = psycopg.connect(url, autocommit=True)
    conn.execute(ESQUEMA)
    return conn


# ------------------------------------------------------------------ trainer
def registrar_entrenamiento(metadata: dict[str, Any], uri_version: str) -> bool:
    conn = _conectar()
    if conn is None:
        return False
    m = metadata["metricas"]
    with conn:
        activa = metadata["resultado"] == "activada"
        if activa:
            conn.execute("UPDATE model_versions SET activa = false")
        conn.execute(
            """INSERT INTO model_versions (version, gcs_uri, f1_train, f1_test, f1_baseline, activa)
               VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (version) DO NOTHING""",
            (metadata["version"], uri_version, m["f1_train"], m["f1_test"],
             m["f1_baseline_test"], activa))
        conn.execute(
            """INSERT INTO trainings (version, inicio, fin, duracion_s, hiperparametros,
                                      mes_corte, resultado, motivo, metadata)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (metadata["version"], metadata["inicio"], metadata["fin"],
             metadata["duracion_entrenamiento_s"], json.dumps(metadata["hiperparametros"]),
             metadata["mes_corte"], metadata["resultado"], metadata["motivo"],
             json.dumps(metadata, ensure_ascii=False)))
    return True


def marcar_activa(version: str) -> None:
    conn = _conectar()
    if conn is None:
        return
    with conn:
        conn.execute("UPDATE model_versions SET activa = (version = %s)", (version,))


# ---------------------------------------------------------------------- API
class EscritorAsincrono:
    """Hilo que vacía una cola de peticiones hacia Postgres."""

    def __init__(self, tamanio: int = 10_000, al_descartar=lambda: None):
        self.cola: queue.Queue = queue.Queue(maxsize=tamanio)
        self.al_descartar = al_descartar
        self.activo = bool(os.environ.get("DATABASE_URL"))
        self._hilo = None

    def iniciar(self) -> None:
        if self.activo and self._hilo is None:
            self._hilo = threading.Thread(target=self._bucle, daemon=True, name="db-writer")
            self._hilo.start()

    def encolar(self, request: dict[str, Any], sugerencias: list[dict[str, Any]]) -> None:
        if not self.activo:
            return
        try:
            self.cola.put_nowait((request, sugerencias))
        except queue.Full:
            self.al_descartar()

    def _bucle(self) -> None:
        conn = None
        while True:
            req, sugs = self.cola.get()
            try:
                conn = conn or _conectar()
                with conn.transaction():
                    fila = conn.execute(
                        """INSERT INTO requests (ts, endpoint, status, latencia_ms, version, pod,
                                                 cliente, mes, n_articulos)
                           VALUES (%(ts)s, %(endpoint)s, %(status)s, %(latencia_ms)s, %(version)s,
                                   %(pod)s, %(cliente)s, %(mes)s, %(n_articulos)s) RETURNING id""",
                        req).fetchone()
                    for s in sugs:
                        conn.execute(
                            """INSERT INTO predictions (request_id, cliente, accesorio, prob, prioridad)
                               VALUES (%s, %s, %s, %s, %s)""",
                            (fila[0], req["cliente"], s["accesorio"], s["prob"], s["prioridad"]))
            except Exception as exc:  # la base caída no tiene que tirar la API
                print(f"db-writer: {exc}", flush=True)
                self.al_descartar()
                conn = None
