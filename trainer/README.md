# trainer · servicio de entrenamiento

Ciclo de vida del modelo de ventas perdidas: **entrenar → evaluar → versionar → guardar → registrar**.
Es el modelo del Trabajo Final del Curso 1 (Random Forest por accesorio) pasado a un módulo de Python.

| Archivo | Qué hace |
|---|---|
| `ventas/datos.py` | Carga el extracto y arma la tabla de compras cliente-mes (Etapa 1 de la notebook) |
| `ventas/modelo.py` | Random Forest por accesorio, regresión logística, línea base, regla de activación y `sugerir()` |
| `ventas/almacen.py` | Lee y escribe en una carpeta local o en `gs://` |
| `train.py` | La corrida completa; guarda `vAAAAMMDD_HHMMSS/model.joblib` + `metadata.json` y actualiza `active.json` |

## Correrlo en local

Los datos no están en el repo: copiá `Export_seco_anonimizado.xlsx` y `productos.xlsx` a `datos/` en la raíz.

```bash
cd trainer
uv run --python 3.12 --with-requirements requirements.txt python train.py \
  --datos ../datos --modelos ../modelos --mes-corte 2026-08
```

- `--mes-corte`: último mes de entrenamiento. El test es el mes siguiente (`--meses-test` para más).
- `--sin-busqueda`: usa los hiperparámetros globales del Curso 1 en vez de la búsqueda por accesorio (de unos 2 minutos a segundos).
- Las mismas opciones se pueden pasar como variables de entorno (`DATA_URI`, `MODELS_URI`, `MES_CORTE`, ...), que es como las recibe el Job de Kubernetes.

Con Docker:

```bash
docker build -t trainer ./trainer
docker run --rm -v "$PWD/datos:/data:ro" -v "$PWD/modelos:/modelos" trainer --mes-corte 2026-08
```

## Verificación contra la notebook

Con el mismo corte que la notebook (`--mes-corte 2026-06 --meses-test 3`) da F1 promedio 0,511, regresión
logística 0,320 y 11 de 20 accesorios confiables: los mismos números que la celda 2.10 y 2.11 del Curso 1.

## Tests

```bash
cd trainer && uv run --python 3.12 --with-requirements requirements.txt --with pytest pytest -q
```
