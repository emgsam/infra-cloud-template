"""Preparación de datos: el mismo camino que la Etapa 1 de la notebook del Curso 1.

facturas (una fila por línea) -> construcción en seco -> artículos con presencia >= 2%
-> producto base -> tabla de compras cliente-mes con 1/0 por producto base.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import pandas as pd

from . import almacen

ARCHIVO_VENTAS = "Export_seco_anonimizado.xlsx"
ARCHIVO_PRODUCTOS = "productos.xlsx"
FAMILIA_SECO = 3
UMBRAL_PRESENCIA = 0.02


@dataclass
class Compras:
    """Tabla de compras lista para el modelo."""

    tabla: pd.DataFrame  # índice compra_id, una columna 1/0 por producto base
    info: pd.DataFrame  # índice compra_id, columnas cliente_id y mes (Period M)
    accesorios: list[str]  # productos base con rol "accesorio": las clases a predecir
    mapa_articulos: dict[str, str]  # articulo_codigo -> producto_base (para la API)


def _leer_excel(ruta: str) -> pd.DataFrame:
    # El Excel del ERP no lo lee openpyxl; calamine sí (igual que en la notebook)
    return pd.read_excel(io.BytesIO(almacen.leer(ruta)), engine="calamine")


def cargar_lineas(origen: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee las líneas de factura y la clasificación de productos (carpeta local o gs://)."""
    lineas = _leer_excel(almacen.unir(origen, ARCHIVO_VENTAS))
    productos = _leer_excel(almacen.unir(origen, ARCHIVO_PRODUCTOS))
    return lineas, productos


def armar_compras(lineas: pd.DataFrame, productos: pd.DataFrame) -> Compras:
    """Pasa de líneas de factura a la tabla de compras cliente-mes (pasos 1.4 a 1.8)."""
    seco = lineas[lineas["familia"] == FAMILIA_SECO].copy()
    if not (seco["cantidad"] > 0).all():
        raise ValueError("hay cantidades en cero o negativas")

    seco["fecha"] = pd.to_datetime(seco["fecha"])
    seco["mes"] = seco["fecha"].dt.to_period("M")
    seco["compra_id"] = seco["cliente_id"].astype(str) + "_" + seco["mes"].astype(str)

    # Artículos presentes en al menos el 2% de las compras
    n_compras = seco["compra_id"].nunique()
    presencia = seco.groupby("articulo_codigo")["compra_id"].nunique() / n_compras
    seleccion = presencia[presencia >= UMBRAL_PRESENCIA].index

    if not productos["articulo_codigo"].is_unique:
        raise ValueError("hay códigos repetidos en productos.xlsx")
    faltan = set(seleccion) - set(productos["articulo_codigo"])
    if faltan:
        raise ValueError(f"artículos seleccionados sin producto base: {sorted(faltan)}")

    seco = seco.merge(productos[["articulo_codigo", "producto_base", "rol"]],
                      on="articulo_codigo", how="left")
    datos = seco[seco["articulo_codigo"].isin(seleccion)]

    tabla = pd.crosstab(datos["compra_id"], datos["producto_base"]).clip(upper=1)
    info = datos.groupby("compra_id")[["cliente_id", "mes"]].first().loc[tabla.index]

    accesorios = sorted(productos.loc[productos["rol"] == "accesorio", "producto_base"].unique())
    # Solo los accesorios que quedaron como columna después de la reducción
    accesorios = [a for a in accesorios if a in tabla.columns]

    mapa = (productos[productos["producto_base"].isin(tabla.columns)]
            .set_index("articulo_codigo")["producto_base"].to_dict())
    return Compras(tabla=tabla, info=info, accesorios=accesorios, mapa_articulos=mapa)


def separar(compras: Compras, mes_corte: str, meses_test: int = 1):
    """Separación temporal: train = meses <= corte; test = los `meses_test` meses siguientes."""
    corte = pd.Period(mes_corte, freq="M")
    meses = compras.info["mes"]
    meses_disponibles = sorted(meses.unique())
    siguientes = [m for m in meses_disponibles if m > corte][:meses_test]
    if len(siguientes) < meses_test:
        raise ValueError(
            f"no hay {meses_test} mes(es) después de {mes_corte} para evaluar; "
            f"el último mes con datos es {meses_disponibles[-1]}"
        )
    train = compras.tabla[meses <= corte]
    test = compras.tabla[meses.isin(siguientes)]
    return train, test, [str(m) for m in siguientes]
