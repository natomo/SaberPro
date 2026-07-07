"""Reducción de dimensionalidad para el pipeline de clustering (UMAP / PCA).

UMAP corre en CPU. Para datasets grandes (>100k filas), se ajusta el
reducer sobre una submuestra de 80k y luego se transforma el resto —
esto es lo que hace viable correr UMAP sobre los 452,020 estudiantes
en Fase 2 sin quedarse sin memoria.
"""

from __future__ import annotations

import numpy as np


def reducir_umap(X: np.ndarray, n_components: int = 2, random_state: int = 42, **kwargs):
    """Reduce la dimensionalidad de X usando UMAP (CPU).

    Args:
        X: Matriz de features preprocesadas.
        n_components: Número de dimensiones del embedding de salida.
        random_state: Semilla aleatoria.
        **kwargs: Parámetros adicionales para umap.UMAP (ej. min_dist).

    Returns:
        Tupla (embedding, reducer_ajustado). El reducer se guarda con joblib
        para poder transformar nuevos datos más adelante.
    """
    import umap.umap_ as umap_cpu  # import diferido: dependencia pesada

    reducer = umap_cpu.UMAP(
        n_components=n_components,
        random_state=random_state,
        n_neighbors=kwargs.pop("n_neighbors", 10),
        low_memory=True,
        n_jobs=-1,
        **kwargs,
    )
    if X.shape[0] > 100_000:
        rng = np.random.default_rng(42)
        idx_fit = rng.choice(X.shape[0], size=80_000, replace=False)
        reducer.fit(X[idx_fit])
        return reducer.transform(X), reducer
    return reducer.fit_transform(X), reducer


def reduce_pca(X: np.ndarray, n_components: float | int = 0.80, random_state: int = 42):
    """PCA usado en el ablation study para justificar el uso de UMAP.

    Args:
        X: Matriz de features preprocesadas.
        n_components: Número de componentes, o fracción de varianza explicada
            a retener (ej. 0.80 = 80% de varianza, como en el ablation study).
        random_state: Semilla aleatoria.

    Returns:
        Tupla (X_pca, pca_ajustado).
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    return X_pca, pca
