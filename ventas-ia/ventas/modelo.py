"""Entrenamiento, líneas base, evaluación y predicción (Etapas 2 y 3 del Curso 1).

Para cada accesorio se entrena un Random Forest que estima, a partir del resto de
los productos de la compra, la probabilidad de que el cliente lo lleve. El modelo
se compara contra dos referencias:
  - regresión logística (el modelo lineal visto en el curso)
  - línea base de frecuencia histórica: "si lleva un producto que en el pasado
    vino acompañado de este accesorio en al menos la mitad de las compras, ofrecerlo"
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid

# Grilla del paso 2.8 de la notebook (3 x 3 x 2 = 18 combinaciones)
GRILLA = {
    "max_depth": [None, 6, 10],
    "min_samples_leaf": [1, 3, 5],
    "class_weight": [None, "balanced"],
}
# Mejor combinación global encontrada en el Curso 1 (se usa con --sin-busqueda)
PARAMS_CURSO1 = {"max_depth": None, "min_samples_leaf": 3, "class_weight": "balanced"}

N_ARBOLES = 200
N_ARBOLES_BUSQUEDA = 100
F1_CONFIABLE = 0.5
UMBRAL_MEDIA = 0.7
UMBRAL_ALTA = 0.8
SEMILLA = 42


# --------------------------------------------------------------------------- métricas
def metricas(y_real, y_pred) -> dict[str, float]:
    return {
        "f1": float(f1_score(y_real, y_pred, zero_division=0)),
        "precision": float(precision_score(y_real, y_pred, zero_division=0)),
        "recall": float(recall_score(y_real, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_real, y_pred)),
    }


def _promedio(por_acc: dict[str, dict[str, float]], clave: str = "f1") -> float:
    return float(np.mean([m[clave] for m in por_acc.values()])) if por_acc else 0.0


# --------------------------------------------------------------------- línea base
def entrenar_linea_base(train: pd.DataFrame, acc: str, minimo: float = 0.5) -> list[str]:
    """Productos que en train vinieron con `acc` en al menos `minimo` de sus compras."""
    X = train.drop(columns=acc)
    con_producto = X.sum()
    juntos = X[train[acc] == 1].sum()
    tasa = (juntos / con_producto.replace(0, np.nan)).fillna(0)
    return tasa[tasa >= minimo].index.tolist()


def predecir_linea_base(disparadores: list[str], X: pd.DataFrame) -> np.ndarray:
    if not disparadores:
        return np.zeros(len(X), dtype=int)
    return (X[disparadores].sum(axis=1) > 0).astype(int).to_numpy()


# ---------------------------------------------------------- búsqueda de hiperparámetros
def buscar_params(sub_train: pd.DataFrame, valid: pd.DataFrame, acc: str) -> dict[str, Any]:
    """Paso 2.10 de la notebook: mejor combinación de la grilla para un accesorio."""
    mejor, mejor_f1 = None, -1.0
    for params in ParameterGrid(GRILLA):
        rf = RandomForestClassifier(n_estimators=N_ARBOLES_BUSQUEDA, random_state=SEMILLA,
                                    n_jobs=-1, **params)
        rf.fit(sub_train.drop(columns=acc), sub_train[acc])
        f1 = f1_score(valid[acc], rf.predict(valid.drop(columns=acc)), zero_division=0)
        if f1 > mejor_f1:  # ante empate queda la primera, como idxmax en la notebook
            mejor, mejor_f1 = params, f1
    return dict(mejor)


# ------------------------------------------------------------------- entrenamiento
def entrenar(train: pd.DataFrame, test: pd.DataFrame, accesorios: list[str],
             validacion: tuple[pd.DataFrame, pd.DataFrame] | None = None,
             log=print) -> dict[str, Any]:
    """Entrena y evalúa todos los accesorios. Devuelve el bundle del modelo y sus métricas.

    validacion = (sub_train, valid): si se pasa, se busca la mejor combinación de la
    grilla para cada accesorio; si no, se usan los parámetros globales del Curso 1.
    """
    inicio = time.perf_counter()
    modelos, params_acc, base_acc = {}, {}, {}
    m_train, m_test, m_logreg, m_base = {}, {}, {}, {}

    for i, acc in enumerate(accesorios, start=1):
        Xtr, ytr = train.drop(columns=acc), train[acc]
        Xte, yte = test.drop(columns=acc), test[acc]

        params = buscar_params(*validacion, acc) if validacion else dict(PARAMS_CURSO1)
        rf = RandomForestClassifier(n_estimators=N_ARBOLES, random_state=SEMILLA,
                                    n_jobs=-1, **params)
        rf.fit(Xtr, ytr)
        modelos[acc], params_acc[acc] = rf, params
        m_train[acc] = metricas(ytr, rf.predict(Xtr))
        m_test[acc] = metricas(yte, rf.predict(Xte))

        logreg = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
        m_logreg[acc] = metricas(yte, logreg.predict(Xte))

        base_acc[acc] = entrenar_linea_base(train, acc)
        m_base[acc] = metricas(yte, predecir_linea_base(base_acc[acc], Xte))
        log(f"[{i:2d}/{len(accesorios)}] {acc:<22} F1 test {m_test[acc]['f1']:.2f}"
            f" | logística {m_logreg[acc]['f1']:.2f} | base {m_base[acc]['f1']:.2f}")

    duracion = time.perf_counter() - inicio
    confiables = [a for a in accesorios if m_test[a]["f1"] >= F1_CONFIABLE]

    bundle = {
        "columnas": list(train.columns),
        "accesorios": list(accesorios),
        "confiables": confiables,
        "modelos": modelos,
        "umbrales": {"media": UMBRAL_MEDIA, "alta": UMBRAL_ALTA},
    }
    resultado = {
        "duracion_entrenamiento_s": round(duracion, 1),
        "hiperparametros": {"n_estimators": N_ARBOLES,
                            "busqueda": "por_accesorio" if validacion else "fija_curso1",
                            "por_accesorio": params_acc},
        "metricas": {
            "f1_train": round(_promedio(m_train), 4),
            "f1_test": round(_promedio(m_test), 4),
            "accuracy_test": round(_promedio(m_test, "accuracy"), 4),
            "f1_logistica_test": round(_promedio(m_logreg), 4),
            "f1_baseline_test": round(_promedio(m_base), 4),
            "por_accesorio": {a: {"train": m_train[a], "test": m_test[a],
                                  "logistica_test": m_logreg[a], "baseline_test": m_base[a]}
                              for a in accesorios},
        },
        "accesorios_confiables": confiables,
    }
    return {"bundle": bundle, "resultado": resultado}


def evaluar_bundle(bundle: dict[str, Any], test: pd.DataFrame) -> float:
    """F1 promedio de un modelo ya guardado sobre otro período (para comparar versiones)."""
    tabla = test.reindex(columns=bundle["columnas"], fill_value=0)
    f1s = []
    for acc in bundle["accesorios"]:
        modelo = bundle["modelos"][acc]
        pred = modelo.predict(tabla.drop(columns=acc))
        f1s.append(f1_score(tabla[acc], pred, zero_division=0))
    return float(np.mean(f1s)) if f1s else 0.0


def decidir(f1_test: float, f1_base: float, f1_anterior: float | None,
            tolerancia: float = 0.01) -> tuple[str, str]:
    """Regla de activación de la propuesta: supera a la línea base y no empeora la anterior."""
    if f1_test <= f1_base:
        return "rechazada", f"F1 {f1_test:.3f} no supera a la línea base {f1_base:.3f}"
    if f1_anterior is not None and f1_test < f1_anterior - tolerancia:
        return "rechazada", (f"F1 {f1_test:.3f} empeora a la versión activa "
                             f"{f1_anterior:.3f} en más de {tolerancia}")
    return "activada", "supera a la línea base y no empeora a la versión activa"


# ------------------------------------------------------------------------ predicción
def sugerir(bundle: dict[str, Any], productos: set[str]) -> list[dict[str, Any]]:
    """Accesorios confiables que probablemente le faltaron a una compra.

    productos: productos base que el cliente llevó en el mes. Devuelve los accesorios
    que NO llevó y cuya probabilidad supera el umbral medio, de mayor a menor.
    """
    columnas = bundle["columnas"]
    fila = pd.DataFrame([[1 if c in productos else 0 for c in columnas]], columns=columnas)
    u = bundle["umbrales"]
    salida = []
    for acc in bundle["confiables"]:
        if acc in productos:
            continue
        modelo = bundle["modelos"][acc]
        if len(modelo.classes_) < 2:  # nunca lo vio en train: no hay nada que sugerir
            continue
        prob = float(modelo.predict_proba(fila.drop(columns=acc))[0, 1])
        if prob >= u["media"]:
            salida.append({"accesorio": acc, "prob": round(prob, 3),
                           "prioridad": "Alta" if prob >= u["alta"] else "Media"})
    return sorted(salida, key=lambda s: s["prob"], reverse=True)
