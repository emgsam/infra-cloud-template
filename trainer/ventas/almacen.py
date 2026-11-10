"""Lectura y escritura de archivos en una carpeta local o en Cloud Storage.

Las rutas que empiezan con gs:// van al bucket (en GKE, con Workload Identity:
sin claves). Cualquier otra ruta es una carpeta local, que es lo que se usa con
Docker Compose y en los tests.
"""

from __future__ import annotations

from pathlib import Path


def _es_gcs(ruta: str) -> bool:
    return ruta.startswith("gs://")


def _blob(ruta: str):
    from google.cloud import storage  # import diferido: local no hace falta

    bucket, _, nombre = ruta[len("gs://"):].partition("/")
    return storage.Client().bucket(bucket).blob(nombre)


def unir(base: str, *partes: str) -> str:
    return "/".join([base.rstrip("/"), *partes])


def leer(ruta: str) -> bytes:
    if _es_gcs(ruta):
        return _blob(ruta).download_as_bytes()
    return Path(ruta).read_bytes()


def escribir(ruta: str, datos: bytes) -> None:
    if _es_gcs(ruta):
        _blob(ruta).upload_from_string(datos)
        return
    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(datos)


def existe(ruta: str) -> bool:
    if _es_gcs(ruta):
        return _blob(ruta).exists()
    return Path(ruta).exists()


def listar_versiones(base_modelos: str) -> list[str]:
    """Nombres de las carpetas vAAAAMMDD_HHMMSS que tienen metadata.json."""
    if _es_gcs(base_modelos):
        from google.cloud import storage

        bucket, _, prefijo = base_modelos[len("gs://"):].partition("/")
        prefijo = prefijo.rstrip("/") + "/"
        nombres = {b.name[len(prefijo):].split("/")[0]
                   for b in storage.Client().list_blobs(bucket, prefix=prefijo)
                   if b.name.endswith("/metadata.json")}
        return sorted(nombres)
    carpeta = Path(base_modelos)
    if not carpeta.exists():
        return []
    return sorted(p.parent.name for p in carpeta.glob("v*/metadata.json"))
