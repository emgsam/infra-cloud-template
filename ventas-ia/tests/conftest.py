import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def datos_sinteticos():
    """6 meses de compras: quien lleva placa casi siempre lleva masilla y tornillos."""
    rng = np.random.default_rng(0)
    productos = pd.DataFrame({
        "articulo_codigo": ["PLA-1", "MON-1", "MAS-1", "TOR-1", "TOR-2", "CIN-1"],
        "producto_base": ["Placa", "Montante", "Masilla", "T2 aguja", "T2 aguja", "Cinta"],
        "rol": ["principal", "principal", "accesorio", "accesorio", "accesorio", "accesorio"],
    })
    filas, factura = [], 0
    for mes in pd.period_range("2026-01", "2026-06", freq="M"):
        for cliente in range(1, 81):
            factura += 1
            fecha = mes.to_timestamp() + pd.Timedelta(days=int(rng.integers(0, 27)))
            lleva_placa = rng.random() < 0.6
            items = []
            if lleva_placa:
                items += ["PLA-1", "MON-1"]
                if rng.random() < 0.85:
                    items.append("MAS-1")
                if rng.random() < 0.8:
                    items.append(rng.choice(["TOR-1", "TOR-2"]))
            else:
                if rng.random() < 0.5:
                    items.append("TOR-1")
                if rng.random() < 0.3:
                    items.append("CIN-1")
            for cod in items or ["CIN-1"]:
                filas.append({"factura_id": factura, "fecha": fecha, "cliente_id": cliente,
                              "articulo_codigo": cod, "articulo_descripcion": cod,
                              "familia": 3, "rubro": 1, "cantidad": 1, "unidad_venta": 1})
    return pd.DataFrame(filas), productos
