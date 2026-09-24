# -*- coding: utf-8 -*-
"""Preprocesamiento — replica exacta de preprocesar_saber_pro() del notebook
del usuario (pipeline_clustering_optimizado_9.ipynb), aplicado a df_maestra.csv."""
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

t0 = time.time()
print("Cargando CSV...")
df = pd.read_csv('/mnt/user-data/uploads/df_maestra.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed|^Column1')]
print(f"Shape original: {df.shape}  ({time.time()-t0:.1f}s)")


def preprocesar_saber_pro(df_raw, sample_n=200_000, random_state=42):
    df_limpio = df_raw.copy()

    mapa_bano = {'1': 1, '2': 2, '3 o 4': 3, '5 o 6': 5, 'MAS DE 6': 6, 'NINGUNA': 0}
    mapa_estrato = {'Sin estrato': 0, 'Estrato 1': 1, 'Estrato 2': 2,
                     'Estrato 3': 3, 'Estrato 4': 4, 'Estrato 5': 5, 'Estrato 6': 6}
    mapa_valormatricula = {
        'Sin costo': 0, 'Menos de 500 mil': 1,
        'Entre 500 mil y menos de 1 millón': 2,
        'Entre 1 millón y menos de 2.5 millones': 3,
        'Entre 2.5 millones y menos de 4 millones': 4,
        'Entre 4 millones y menos de 5.5 millones': 5,
        'Entre 5.5 millones y menos de 7 millones': 6,
        'Más de 7 millones': 7}
    mapa_educ = {
        'Ninguno': 0, 'Primaria incompleta': 1, 'Primaria completa': 2,
        'Secundaria (Bachillerato) incompleta': 3,
        'Secundaria (Bachillerato) completa': 4,
        'Técnica o tecnológica incompleta': 5,
        'Técnica o tecnológica completa': 6,
        'Educación profesional incompleta': 7,
        'EDUCACIÓN PROFESIONAL COMPLETA': 8, 'POSTGRADO': 9}
    mapeo_horas = {'0': 0, 'Menos de 10 horas': 1, 'Entre 11 y 20 horas': 2,
                    'Entre 21 y 30 horas': 3, 'Más de 30 horas': 4}
    mapeo_semestre = {str(i).zfill(2): i for i in range(1, 12)}
    mapeo_semestre['12 o más'] = 12

    mapeables = {
        'FAMI_CUANTOSCOMPARTEBAÑO':      mapa_bano,
        'FAMI_ESTRATOVIVIENDA':          mapa_estrato,
        'ESTU_VALORMATRICULAUNIVERSIDAD': mapa_valormatricula,
        'FAMI_EDUCACIONPADRE':           mapa_educ,
        'FAMI_EDUCACIONMADRE':           mapa_educ,
        'ESTU_HORASSEMANATRABAJA':       mapeo_horas,
        'ESTU_SEMESTRECURSA':            mapeo_semestre,
    }
    for col, mapa in mapeables.items():
        if col in df_limpio.columns:
            df_limpio[col] = df_limpio[col].map(mapa)

    columnas_puntaje = [
        'MOD_RAZONA_CUANTITAT_PUNT', 'MOD_LECTURA_CRITICA_PUNT',
        'MOD_COMPETEN_CIUDADA_PUNT', 'MOD_INGLES_PUNT', 'MOD_COMUNI_ESCRITA_PUNT']
    columnas_ordinales = [
        'FAMI_ESTRATOVIVIENDA', 'ESTU_VALORMATRICULAUNIVERSIDAD',
        'FAMI_EDUCACIONPADRE', 'FAMI_EDUCACIONMADRE', 'ESTU_HORASSEMANATRABAJA']
    columnas_nominales = [
        'ESTU_TITULOOBTENIDOBACHILLER',
        'ESTU_PAGOMATRICULABECA', 'ESTU_PAGOMATRICULACREDITO',
        'ESTU_PAGOMATRICULAPADRES', 'ESTU_PAGOMATRICULAPROPIO',
        'ESTU_COMOCAPACITOEXAMENSB11',
        'FAMI_TIENEINTERNET', 'FAMI_TIENECOMPUTADOR',
        'FAMI_TIENEAUTOMOVIL', 'FAMI_TIENELAVADORA']
    col_geo = 'ESTU_COD_DEPTO_PRESENTACION'

    for col in columnas_puntaje:
        if col in df_limpio.columns:
            df_limpio[col] = pd.to_numeric(df_limpio[col], errors='coerce')
            df_limpio[col] = df_limpio[col].fillna(df_limpio[col].mean())
    for col in columnas_ordinales:
        if col in df_limpio.columns:
            df_limpio[col] = df_limpio[col].fillna(df_limpio[col].median())
    for col in columnas_nominales:
        if col in df_limpio.columns:
            df_limpio[col] = df_limpio[col].fillna(df_limpio[col].mode(dropna=True)[0])

    cols_usar = columnas_ordinales + columnas_puntaje + columnas_nominales
    # columnas extra que necesitamos conservar para los reanálisis (no entran a X)
    cols_extra = [col_geo, 'INST_COD_INSTITUCION', 'ESTU_PRGM_ACADEMICO',
                  'PERIODO', 'PUNT_GLOBAL', 'ESTU_CONSECUTIVO']
    cols_df = cols_usar + [c for c in cols_extra if c in df_limpio.columns]
    df_filtrado = df_limpio[[c for c in cols_df if c in df_limpio.columns]].copy()
    df_filtrado = df_filtrado.dropna(subset=[c for c in columnas_puntaje if c in df_filtrado.columns])
    print(f"Filas después de limpieza: {len(df_filtrado):,}")

    cols_punt = [c for c in columnas_puntaje if c in df_filtrado.columns]
    cols_ord = [c for c in columnas_ordinales if c in df_filtrado.columns]
    cols_nom = [c for c in columnas_nominales if c in df_filtrado.columns]

    scaler = StandardScaler()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_punt = scaler.fit_transform(df_filtrado[cols_punt])
    X_ohe = encoder.fit_transform(df_filtrado[cols_nom])
    X = np.hstack([df_filtrado[cols_ord].values, X_punt, X_ohe])
    feature_names = (cols_ord + list(scaler.get_feature_names_out(cols_punt))
                      + list(encoder.get_feature_names_out(cols_nom)))

    if np.isnan(X).any():
        from sklearn.impute import SimpleImputer
        X = SimpleImputer(strategy='median').fit_transform(X)
        print("NaN residuales imputados con mediana")

    if sample_n is not None and sample_n < df_filtrado.shape[0]:
        rng = np.random.default_rng(seed=random_state)
        idx = rng.choice(df_filtrado.shape[0], size=sample_n, replace=False)
        df_filtrado = df_filtrado.iloc[idx].reset_index(drop=True)
        X = X[idx]

    print(f"Preprocesamiento completo — shape X: {X.shape}")
    return df_limpio, df_filtrado, X, feature_names, encoder, scaler


if __name__ == "__main__":
    t1 = time.time()
    df_limpio, df_filtrado_full, X_full, feature_names, encoder, scaler = \
        preprocesar_saber_pro(df, sample_n=None)
    print(f"Preprocesamiento full: {time.time()-t1:.1f}s")
    print("N final (full):", X_full.shape)
    np.save('/home/claude/X_full.npy', X_full)
    df_filtrado_full.to_pickle('/home/claude/df_filtrado_full.pkl')
    with open('/home/claude/feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_names))
    print("Guardado X_full.npy, df_filtrado_full.parquet, feature_names.txt")
    print(f"TOTAL: {time.time()-t0:.1f}s")
