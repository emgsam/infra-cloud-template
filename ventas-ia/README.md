# ventas-ia · servicio de IA de ventas perdidas

Trabajo Final del Curso 2. Es el modelo del Trabajo Final del Curso 1 (Random Forest por accesorio)
convertido en dos servicios que comparten el paquete `ventas/`:

| Pieza | Imagen | Qué hace |
|---|---|---|
| `train.py` | `Dockerfile.trainer` | Ciclo de vida: entrenar → evaluar → versionar → guardar → registrar. Corre como Job |
| `api.py` | `Dockerfile.api` | FastAPI: `/predict`, `/healthz`, `/metrics`, `/versions`, `/versions/{v}/activate` |
| `ventas/datos.py` | | Carga el extracto y arma la tabla de compras cliente-mes (Etapa 1 de la notebook) |
| `ventas/modelo.py` | | Random Forest por accesorio, regresión logística, línea base, regla de activación, `sugerir()` |
| `ventas/almacen.py` | | Lee y escribe en una carpeta local o en `gs://` |
| `ventas/registro.py` | | Tablas de Postgres y escritura asíncrona desde la API |

## Cómo se versiona

Cada corrida del trainer guarda `<modelos>/vAAAAMMDD_HHMMSS/model.joblib` y `metadata.json`
(métricas en train y test, línea base, regresión logística, versión anterior, hiperparámetros por
accesorio, duración, versiones de librerías y SHA de git). Si la versión **supera a la línea base y su
F1 no cae más de 0,01 respecto de la activa** (re-evaluada sobre el mismo mes), se escribe
`<modelos>/active.json`. La API lee ese puntero al arrancar y cada `RELOAD_SECONDS`, así todas las
réplicas sirven la misma versión.

## Correrlo en local

Los datos no están en el repo: copiá `Export_seco_anonimizado.xlsx` y `productos.xlsx` a `datos/` en la raíz.

Con Docker Compose (Postgres + trainer + API):

```bash
cd ventas-ia
docker compose up -d postgres
docker compose run --rm trainer --mes-corte 2026-08    # unos 2 minutos
docker compose up -d api                               # http://localhost:8000/docs
```

Sin Docker:

```bash
cd ventas-ia
uv run --python 3.12 --with-requirements requirements-api.txt python train.py \
  --datos ../datos --modelos ./modelos --mes-corte 2026-08
MODELS_URI=./modelos uv run --python 3.12 --with-requirements requirements-api.txt python api.py
```

Opciones del trainer (también como variables de entorno `DATA_URI`, `MODELS_URI`, `MES_CORTE`, `MESES_TEST`, `SIN_BUSQUEDA`):

- `--mes-corte`: último mes de entrenamiento. El test es el mes siguiente (`--meses-test` para más).
- `--sin-busqueda`: usa los hiperparámetros globales del Curso 1 en vez de buscar por accesorio (de 2 minutos a 20 segundos).
- `DATABASE_URL`: si está, registra la corrida en `trainings` y `model_versions`.

## Ejemplo de `/predict`

```bash
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "cliente": 1001, "mes": "2026-09",
  "productos": ["Placa STD 12,5", "Montante 34", "Solera 35", "T2 aguja", "Masilla LPU"]}'
```

```json
{"version": "v20261110_174649", "cliente": 1001, "mes": "2026-09",
 "productos_reconocidos": ["Masilla LPU", "Montante 34", "Placa STD 12,5", "Solera 35", "T2 aguja"],
 "no_reconocidos": [],
 "sugerencias": [{"accesorio": "T1 aguja", "prob": 0.747, "prioridad": "Media"},
                 {"accesorio": "Cantonera metálica", "prob": 0.716, "prioridad": "Media"}],
 "latencia_ms": 52.8}
```

También acepta `"articulos": [{"codigo": "DUR-319093", "cantidad": 10}]` con los códigos del ERP.

## Verificación contra la notebook

Con el mismo corte que la notebook (`--mes-corte 2026-06 --meses-test 3`) da F1 promedio 0,511,
regresión logística 0,320 y 11 de 20 accesorios confiables: los mismos números que los pasos 2.10 y
2.11 del Curso 1.

## Tests

```bash
cd ventas-ia && uv run --python 3.12 --with-requirements requirements-api.txt --with pytest --with httpx pytest -q
```
