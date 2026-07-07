"""Algoritmos de clustering: K-Means, GMM, DBSCAN y HDBSCAN.

Incluye tanto wrappers simples (usados en tests y en Fase 1 vía
hyperparameter_search) como la función entrenar_modelo_final que
replica exactamente la Fase 2 del pipeline original: entrena el
algoritmo ganador de la búsqueda de hiperparámetros sobre los 452k
estudiantes completos.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture


def run_kmeans(X: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Ejecuta K-Means según los hiperparámetros de config.yaml (uso en tests)."""
    params = config["clustering"]["algorithms"]["kmeans"]
    model = KMeans(
        n_clusters=params["n_clusters"],
        n_init=params["n_init"],
        random_state=config["random_state"],
    )
    return model.fit_predict(X)


def run_gmm(X: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Ejecuta Gaussian Mixture Model según config.yaml."""
    params = config["clustering"]["algorithms"]["gmm"]
    model = GaussianMixture(
        n_components=params["n_components"],
        covariance_type=params["covariance_type"],
        random_state=config["random_state"],
    )
    return model.fit_predict(X)


def run_dbscan(X: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Ejecuta DBSCAN según config.yaml."""
    params = config["clustering"]["algorithms"]["dbscan"]
    model = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
    return model.fit_predict(X)


def run_hdbscan(X: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Ejecuta HDBSCAN según config.yaml."""
    import hdbscan  # import diferido: dependencia pesada

    params = config["clustering"]["algorithms"]["hdbscan"]
    model = hdbscan.HDBSCAN(
        min_cluster_size=params["min_cluster_size"],
        min_samples=params["min_samples"],
    )
    return model.fit_predict(X)


def entrenar_modelo_final(
    X_umap_full: np.ndarray,
    best_algo: str,
    best_k: int | None = None,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 10,
    hdbscan_mc: int = 10,
    hdbscan_ms: int = 5,
):
    """Entrena el modelo final de Fase 2 sobre los 452k estudiantes completos.

    Replica el bloque de dispatch de la sección 12 del notebook original:
    elige el algoritmo según best_algo y usa los hiperparámetros ganadores
    de la búsqueda de Fase 1.

    Args:
        X_umap_full: Embedding UMAP calculado sobre los datos completos.
        best_algo: Uno de 'KMeans', 'GMM', 'DBSCAN', 'HDBSCAN'.
        best_k: Número de clústeres (K-Means/GMM). Requerido para esos dos.
        dbscan_eps, dbscan_min_samples: Hiperparámetros de DBSCAN.
        hdbscan_mc, hdbscan_ms: min_cluster_size y min_samples de HDBSCAN.

    Returns:
        Tupla (modelo_ajustado_o_None, labels_final). El modelo es None para
        DBSCAN/HDBSCAN (no tienen atributo reutilizable equivalente a
        KMeans/GMM más allá de fit_predict).
    """
    if best_algo == "KMeans":
        km = MiniBatchKMeans(n_clusters=best_k, random_state=42, n_init="auto", batch_size=10_000)
        labels_final = km.fit_predict(X_umap_full)
        return km, labels_final

    elif best_algo == "GMM":
        gmm = GaussianMixture(n_components=best_k, random_state=42, covariance_type="full", n_init=1)
        labels_final = gmm.fit_predict(X_umap_full)
        return gmm, labels_final

    elif best_algo == "DBSCAN":
        labels_final = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, n_jobs=-1).fit_predict(
            X_umap_full
        )
        return None, labels_final

    elif best_algo == "HDBSCAN":
        import hdbscan

        labels_final = hdbscan.HDBSCAN(min_cluster_size=hdbscan_mc, min_samples=hdbscan_ms).fit_predict(
            X_umap_full
        )
        return None, labels_final

    raise ValueError(f"Algoritmo no reconocido: {best_algo}")
