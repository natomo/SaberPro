"""Pruebas unitarias del pipeline de clustering Saber Pro.

Ejecutar con: pytest tests/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saberpro_clustering.clustering import entrenar_modelo_final  # noqa: E402
from saberpro_clustering.evaluation import calcular_metricas, dunn_index, validar_representatividad  # noqa: E402
from saberpro_clustering.preprocessing import preprocesar_saber_pro  # noqa: E402


@pytest.fixture
def toy_embedding() -> np.ndarray:
    rng = np.random.default_rng(0)
    cluster_1 = rng.normal(loc=(0, 0), scale=0.3, size=(50, 2))
    cluster_2 = rng.normal(loc=(5, 5), scale=0.3, size=(50, 2))
    cluster_3 = rng.normal(loc=(0, 5), scale=0.3, size=(50, 2))
    return np.vstack([cluster_1, cluster_2, cluster_3])


@pytest.fixture
def toy_raw_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 300
    return pd.DataFrame(
        {
            "MOD_RAZONA_CUANTITAT_PUNT": rng.normal(150, 20, n),
            "MOD_LECTURA_CRITICA_PUNT": rng.normal(150, 20, n),
            "MOD_COMPETEN_CIUDADA_PUNT": rng.normal(150, 20, n),
            "MOD_INGLES_PUNT": rng.normal(150, 20, n),
            "MOD_COMUNI_ESCRITA_PUNT": rng.normal(150, 20, n),
            "FAMI_ESTRATOVIVIENDA": rng.choice(["Estrato 1", "Estrato 2", "Estrato 3"], n),
            "ESTU_VALORMATRICULAUNIVERSIDAD": rng.choice(["Sin costo", "Menos de 500 mil"], n),
            "FAMI_EDUCACIONPADRE": rng.choice(["Ninguno", "Primaria completa", "POSTGRADO"], n),
            "FAMI_EDUCACIONMADRE": rng.choice(["Ninguno", "Primaria completa", "POSTGRADO"], n),
            "ESTU_HORASSEMANATRABAJA": rng.choice(["0", "Menos de 10 horas"], n),
            "ESTU_TITULOOBTENIDOBACHILLER": rng.choice(["Bachiller académico"], n),
            "ESTU_PAGOMATRICULABECA": rng.choice(["Si", "No"], n),
            "ESTU_PAGOMATRICULACREDITO": rng.choice(["Si", "No"], n),
            "ESTU_PAGOMATRICULAPADRES": rng.choice(["Si", "No"], n),
            "ESTU_PAGOMATRICULAPROPIO": rng.choice(["Si", "No"], n),
            "ESTU_COMOCAPACITOEXAMENSB11": rng.choice(["Por internet"], n),
            "FAMI_TIENEINTERNET": rng.choice(["Si", "No"], n),
            "FAMI_TIENECOMPUTADOR": rng.choice(["Si", "No"], n),
            "FAMI_TIENEAUTOMOVIL": rng.choice(["Si", "No"], n),
            "FAMI_TIENELAVADORA": rng.choice(["Si", "No"], n),
            "ESTU_COD_DEPTO_PRESENTACION": rng.choice([5, 11, 66], n),
        }
    )


def test_preprocesar_saber_pro_shapes(toy_raw_df):
    df_limpio, df_filtrado, X, feature_names, encoder, scaler = preprocesar_saber_pro(
        toy_raw_df, sample_n=None
    )
    assert len(df_filtrado) == len(toy_raw_df)
    assert X.shape[0] == len(df_filtrado)
    assert len(feature_names) == X.shape[1]
    assert not np.isnan(X).any()


def test_preprocesar_saber_pro_sampling(toy_raw_df):
    _, df_filtrado, X, *_ = preprocesar_saber_pro(toy_raw_df, sample_n=100, random_state=42)
    assert len(df_filtrado) == 100
    assert X.shape[0] == 100


def test_entrenar_modelo_final_kmeans(toy_embedding):
    modelo, labels = entrenar_modelo_final(toy_embedding, best_algo="KMeans", best_k=3)
    assert modelo is not None
    assert len(set(labels)) == 3
    assert len(labels) == len(toy_embedding)


def test_entrenar_modelo_final_gmm(toy_embedding):
    modelo, labels = entrenar_modelo_final(toy_embedding, best_algo="GMM", best_k=3)
    assert modelo is not None
    assert len(set(labels)) == 3


def test_entrenar_modelo_final_unknown_algo(toy_embedding):
    with pytest.raises(ValueError):
        entrenar_modelo_final(toy_embedding, best_algo="NoExiste")


def test_calcular_metricas_returns_expected_keys(toy_embedding):
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(n_clusters=3, random_state=42, n_init="auto")
    labels = km.fit_predict(toy_embedding)
    m = calcular_metricas(toy_embedding, labels, inertia=km.inertia_)
    assert m is not None
    assert set(m.keys()) == {
        "silhouette", "calinski", "davies_bouldin", "dunn",
        "inertia", "bic", "n_clusters", "n_ruido",
    }


def test_calcular_metricas_handles_single_cluster(toy_embedding):
    labels = np.zeros(len(toy_embedding))
    m = calcular_metricas(toy_embedding, labels)
    assert m is None


def test_dunn_index_two_separated_clusters(toy_embedding):
    labels = np.array([0] * 50 + [1] * 50 + [2] * 50)
    d = dunn_index(toy_embedding, labels)
    assert d > 0


def test_validar_representatividad_identical_samples(toy_raw_df):
    df_rep = validar_representatividad(toy_raw_df, toy_raw_df, vars_validar={"MOD_INGLES_PUNT": "continua"})
    assert len(df_rep) == 1
    assert df_rep.iloc[0]["diferencia_pct"] == 0
    assert df_rep.iloc[0]["representa"] == "✅"
