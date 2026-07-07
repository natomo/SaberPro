"""Limpieza, imputación, codificación y muestreo de variables Saber Pro.

Migrado directamente del pipeline original (Colab). Contiene los mapeos
ordinales exactos usados para el artículo — no modificar los valores de
los diccionarios sin volver a validar las métricas reportadas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Mapeos ordinales — igual que el notebook original ──────────────────
MAPA_BANO = {
    "1": 1, "2": 2, "3 o 4": 3, "5 o 6": 5, "MAS DE 6": 6, "NINGUNA": 0,
}
MAPA_ESTRATO = {
    "Sin estrato": 0, "Sin Estrato": 0, "Estrato 1": 1, "Estrato 2": 2,
    "Estrato 3": 3, "Estrato 4": 4, "Estrato 5": 5, "Estrato 6": 6,
}
MAPA_VALORMATRICULA = {
    "Sin costo": 0, "No pagó matrícula": 0,
    "Menos de 500 mil": 1,
    "Entre 500 mil y menos de 1 millón": 2,
    "Entre 1 millón y menos de 2.5 millones": 3,
    "Entre 2.5 millones y menos de 4 millones": 4,
    "Entre 4 millones y menos de 5.5 millones": 5,
    "Entre 5.5 millones y menos de 7 millones": 6,
    "Más de 7 millones": 7,
}
MAPA_EDUC = {
    "Ninguno": 0, "Primaria incompleta": 1, "Primaria completa": 2,
    "Secundaria (Bachillerato) incompleta": 3,
    "Secundaria (Bachillerato) completa": 4,
    "Técnica o tecnológica incompleta": 5,
    "Técnica o tecnológica completa": 6,
    "Educación profesional incompleta": 7,
    "EDUCACIÓN PROFESIONAL COMPLETA": 8, "Educación profesional completa": 8,
    "POSTGRADO": 9, "Postgrado": 9,
    # "No sabe" / "No Aplica" quedan sin mapear (NaN) — se imputan con la
    # mediana más adelante, igual que cualquier otro valor faltante genuino.
}
MAPEO_HORAS = {
    "0": 0, "Menos de 10 horas": 1, "Entre 11 y 20 horas": 2,
    "Entre 21 y 30 horas": 3, "Más de 30 horas": 4,
}
MAPEO_SEMESTRE = {str(i).zfill(2): i for i in range(1, 12)}
MAPEO_SEMESTRE["12 o más"] = 12

MAPEABLES = {
    "FAMI_CUANTOSCOMPARTEBAÑO": MAPA_BANO,
    "FAMI_ESTRATOVIVIENDA": MAPA_ESTRATO,
    "ESTU_VALORMATRICULAUNIVERSIDAD": MAPA_VALORMATRICULA,
    "FAMI_EDUCACIONPADRE": MAPA_EDUC,
    "FAMI_EDUCACIONMADRE": MAPA_EDUC,
    "ESTU_HORASSEMANATRABAJA": MAPEO_HORAS,
    "ESTU_SEMESTRECURSA": MAPEO_SEMESTRE,
}

# ── Columnas por tipo ────────────────────────────────────────────────────
COLUMNAS_PUNTAJE = [
    "MOD_RAZONA_CUANTITAT_PUNT", "MOD_LECTURA_CRITICA_PUNT",
    "MOD_COMPETEN_CIUDADA_PUNT", "MOD_INGLES_PUNT",
    "MOD_COMUNI_ESCRITA_PUNT",
]
COLUMNAS_ORDINALES = [
    "FAMI_ESTRATOVIVIENDA", "ESTU_VALORMATRICULAUNIVERSIDAD",
    "FAMI_EDUCACIONPADRE", "FAMI_EDUCACIONMADRE",
    "ESTU_HORASSEMANATRABAJA",
]
COLUMNAS_NOMINALES = [
    "ESTU_TITULOOBTENIDOBACHILLER",
    "ESTU_PAGOMATRICULABECA", "ESTU_PAGOMATRICULACREDITO",
    "ESTU_PAGOMATRICULAPADRES", "ESTU_PAGOMATRICULAPROPIO",
    "ESTU_COMOCAPACITOEXAMENSB11",
    "FAMI_TIENEINTERNET", "FAMI_TIENECOMPUTADOR",
    "FAMI_TIENEAUTOMOVIL", "FAMI_TIENELAVADORA",
]
COL_GEO = "ESTU_COD_DEPTO_PRESENTACION"


def load_raw_data(path: str) -> pd.DataFrame:
    """Carga el dataset crudo de Saber Pro (csv o parquet).

    Args:
        path: Ruta al archivo de datos (local o de Drive montado en Colab).

    Returns:
        DataFrame con los datos crudos.
    """
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def preprocesar_saber_pro(
    df_raw: pd.DataFrame, sample_n: int | None = 200_000, random_state: int = 42
):
    """Limpieza, imputación, codificación y muestreo del dataset Saber Pro.

    Args:
        df_raw: DataFrame crudo (salida de load_raw_data).
        sample_n: Tamaño de muestra a extraer (None para usar todos los datos,
            usado en Fase 2 con los 452,020 estudiantes completos).
        random_state: Semilla para el muestreo aleatorio.

    Returns:
        Tupla (df_limpio, df_filtrado, X, feature_names, encoder, scaler):
            - df_limpio: DataFrame completo con mapeos aplicados (sin filtrar).
            - df_filtrado: DataFrame filtrado y muestreado usado en clustering.
            - X: matriz de features (ordinales + puntajes escalados + one-hot).
            - feature_names: nombres de columnas de X.
            - encoder, scaler: objetos ajustados (para guardar con joblib).
    """
    df_limpio = df_raw.copy()

    # Mapear directo sin _upper() — los valores del CSV ya tienen
    # la capitalización correcta para estas columnas
    for col, mapa in MAPEABLES.items():
        if col in df_limpio.columns:
            df_limpio[col] = df_limpio[col].map(mapa)

    # Verificar mapeos
    for col in MAPEABLES:
        if col in df_limpio.columns:
            n_nan = df_limpio[col].isna().sum()
            pct = n_nan / len(df_limpio) * 100
            status = "✅" if pct < 5 else "⚠️ "
            print(f"  {status} {col}: {n_nan:,} NaN ({pct:.1f}%)")

    # ── Imputación antes de construir df_filtrado ─────────────
    for col in COLUMNAS_PUNTAJE:
        if col in df_limpio.columns:
            df_limpio[col] = pd.to_numeric(df_limpio[col], errors="coerce")
            df_limpio[col] = df_limpio[col].fillna(df_limpio[col].mean())
    for col in COLUMNAS_ORDINALES:
        if col in df_limpio.columns:
            df_limpio[col] = df_limpio[col].fillna(df_limpio[col].median())
    for col in COLUMNAS_NOMINALES:
        if col in df_limpio.columns:
            df_limpio[col] = df_limpio[col].fillna(df_limpio[col].mode(dropna=True)[0])

    cols_usar = COLUMNAS_ORDINALES + COLUMNAS_PUNTAJE + COLUMNAS_NOMINALES
    cols_df = cols_usar + ([COL_GEO] if COL_GEO in df_limpio.columns else [])
    df_filtrado = df_limpio[[c for c in cols_df if c in df_limpio.columns]].copy()
    df_filtrado = df_filtrado.dropna(subset=COLUMNAS_PUNTAJE)
    print(f"\n✅ Filas después de limpieza: {len(df_filtrado):,}")

    # ── Escalado y codificación ───────────────────────────────
    cols_punt = [c for c in COLUMNAS_PUNTAJE if c in df_filtrado.columns]
    cols_ord = [c for c in COLUMNAS_ORDINALES if c in df_filtrado.columns]
    cols_nom = [c for c in COLUMNAS_NOMINALES if c in df_filtrado.columns]

    scaler = StandardScaler()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    X_punt = scaler.fit_transform(df_filtrado[cols_punt])
    X_ohe = encoder.fit_transform(df_filtrado[cols_nom])
    X = np.hstack([df_filtrado[cols_ord].values, X_punt, X_ohe])

    feature_names = (
        cols_ord
        + list(scaler.get_feature_names_out(cols_punt))
        + list(encoder.get_feature_names_out(cols_nom))
    )

    # Seguridad: imputar NaN residuales en X (valores no cubiertos por mapas)
    if np.isnan(X).any():
        from sklearn.impute import SimpleImputer

        X = SimpleImputer(strategy="median").fit_transform(X)
        print("⚠️  NaN residuales imputados con mediana")

    # ── Muestreo ──────────────────────────────────────────────
    if sample_n is not None and sample_n < df_filtrado.shape[0]:
        rng = np.random.default_rng(seed=random_state)
        idx = rng.choice(df_filtrado.shape[0], size=sample_n, replace=False)
        df_filtrado = df_filtrado.iloc[idx].reset_index(drop=True)
        X = X[idx]

    print(f"✅ Preprocesamiento completo — shape X: {X.shape}")
    return df_limpio, df_filtrado, X, feature_names, encoder, scaler
