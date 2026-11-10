import pytest
from fastapi.testclient import TestClient

import train as cli


@pytest.fixture
def cliente(datos_sinteticos, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "cargar_lineas", lambda origen: datos_sinteticos)
    modelos = tmp_path / "modelos"
    base = ["--datos", "x", "--modelos", str(modelos), "--sin-busqueda"]
    cli.main([*base, "--mes-corte", "2026-04", "--version", "v1"])
    cli.main([*base, "--mes-corte", "2026-05", "--version", "v2"])
    monkeypatch.setenv("MODELS_URI", str(modelos))
    monkeypatch.setenv("RELOAD_SECONDS", "0")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import api
    api._estado.update(bundle=None, version=None)
    with TestClient(api.app) as c:
        yield c


def test_healthz_y_metrics(cliente):
    r = cliente.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert "ventas_modelo_info" in cliente.get("/metrics").text


def test_predict_por_codigo(cliente):
    r = cliente.post("/predict", json={"cliente": 7, "mes": "2026-06",
                                       "articulos": [{"codigo": "PLA-1", "cantidad": 40},
                                                     {"codigo": "MON-1"}, {"codigo": "XXX"}]})
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["productos_reconocidos"] == ["Montante", "Placa"]
    assert cuerpo["no_reconocidos"] == ["XXX"]
    assert {s["accesorio"] for s in cuerpo["sugerencias"]} <= {"Masilla", "T2 aguja", "Cinta"}
    assert cuerpo["sugerencias"], "con placa y montante tendría que sugerir masilla o tornillos"


def test_predict_valida_entrada(cliente):
    assert cliente.post("/predict", json={}).status_code == 422
    assert cliente.post("/predict", json={"mes": "sept", "productos": ["Placa"]}).status_code == 422


def test_versiones_y_activacion(cliente):
    r = cliente.get("/versions").json()
    assert [v["version"] for v in r["versiones"]] == ["v1", "v2"]
    assert cliente.post("/versions/v9/activate").status_code == 404
    assert cliente.post("/versions/v1/activate").json()["activa"] == "v1"
    assert cliente.get("/healthz").json()["version"] == "v1"
    assert cliente.get("/versions").json()["activa"] == "v1"
