import json

import pandas as pd
import pytest

import train as cli
from ventas import datos as datos_mod
from ventas.datos import armar_compras, separar
from ventas.modelo import decidir, entrenar, sugerir


def test_tabla_de_compras(datos_sinteticos):
    lineas, productos = datos_sinteticos
    compras = armar_compras(lineas, productos)
    # una fila por cliente-mes, columnas 1/0 por producto base (T2 aguja agrupa 2 códigos)
    assert set(compras.tabla.columns) == {"Placa", "Montante", "Masilla", "T2 aguja", "Cinta"}
    assert compras.tabla.isin([0, 1]).all().all()
    assert compras.accesorios == ["Cinta", "Masilla", "T2 aguja"]
    assert compras.mapa_articulos["TOR-2"] == "T2 aguja"


def test_separacion_temporal(datos_sinteticos):
    compras = armar_compras(*datos_sinteticos)
    train, test, meses = separar(compras, "2026-04", meses_test=1)
    assert meses == ["2026-05"]
    assert compras.info.loc[train.index, "mes"].max() == pd.Period("2026-04", "M")
    with pytest.raises(ValueError, match="no hay 1 mes"):
        separar(compras, "2026-06")


def test_entrena_y_sugiere(datos_sinteticos):
    compras = armar_compras(*datos_sinteticos)
    train, test, _ = separar(compras, "2026-05")
    salida = entrenar(train, test, compras.accesorios, log=lambda *_: None)
    m = salida["resultado"]["metricas"]
    assert m["por_accesorio"]["Masilla"]["test"]["f1"] > 0.8
    assert "Masilla" in salida["bundle"]["confiables"]
    sugerencias = sugerir(salida["bundle"], {"Placa", "Montante"})
    assert sugerencias[0]["accesorio"] in {"Masilla", "T2 aguja"}
    assert sugerencias[0]["prioridad"] in {"Alta", "Media"}
    # lo que ya llevó no se sugiere
    assert all(s["accesorio"] != "Masilla" for s in sugerir(salida["bundle"], {"Placa", "Masilla"}))


@pytest.mark.parametrize("f1,base,anterior,esperado", [
    (0.60, 0.40, None, "activada"),
    (0.30, 0.40, None, "rechazada"),
    (0.595, 0.40, 0.60, "activada"),   # cae menos de 0,01: se tolera
    (0.55, 0.40, 0.60, "rechazada"),
])
def test_regla_de_activacion(f1, base, anterior, esperado):
    assert decidir(f1, base, anterior)[0] == esperado


def test_ciclo_completo_versiona_y_activa(datos_sinteticos, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "cargar_lineas", lambda origen: datos_sinteticos)
    modelos = tmp_path / "modelos"
    args = ["--datos", "x", "--modelos", str(modelos), "--sin-busqueda"]

    m1 = cli.main([*args, "--mes-corte", "2026-04", "--version", "v1"])
    assert m1["resultado"] == "activada"
    assert json.loads((modelos / "active.json").read_text())["version"] == "v1"
    meta = json.loads((modelos / "v1" / "metadata.json").read_text())
    assert meta["meses_test"] == ["2026-05"] and "scikit-learn" in meta["librerias"]

    # la segunda versión se compara con la primera sobre su mismo mes de test
    m2 = cli.main([*args, "--mes-corte", "2026-05", "--version", "v2"])
    assert m2["version_anterior"] == "v1"
    assert m2["f1_version_anterior_test"] is not None
    assert (modelos / "v2" / "model.joblib").exists()
