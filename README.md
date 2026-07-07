# Clustering no supervisado de estudiantes Saber Pro (Colombia)

Repositorio de código complementario al artículo de investigación sobre segmentación no supervisada de **452,020 estudiantes** que presentaron la prueba **Saber Pro** en Colombia, mediante reducción de dimensionalidad (UMAP) y algoritmos de clustering (K-Means, Gaussian Mixture Models, DBSCAN y HDBSCAN).

> **Estado:** en preparación para publicación. Este README se actualizará con la referencia completa (DOI, autores, revista) una vez el artículo sea aceptado.

## Autoría

- Nathalia Orozco Morales — UNIMINUTO, Grupo de Investigación INTI
- Leonardo Valderrama García
- Marisol Jiménez
- Cristian Rincón-Guio

## Resumen del proyecto

El proyecto identifica perfiles latentes de desempeño académico a partir de los resultados de Saber Pro, explorando asimetrías asociadas a la educación de los padres y otras variables sociodemográficas. Se compara la calidad y estabilidad de distintos algoritmos de clustering (K-Means, GMM, DBSCAN, HDBSCAN) sobre representaciones de baja dimensionalidad obtenidas con UMAP.

### Estrategia en dos fases

- **Fase 1 — búsqueda de hiperparámetros** sobre una muestra de 200,000 estudiantes: grid search de algoritmo × dimensión UMAP × parámetros propios de cada método, evaluado con silhouette, Calinski-Harabasz, Davies-Bouldin y el índice de Dunn.
- **Fase 2 — modelo final** entrenado sobre los 452,020 estudiantes completos con la configuración ganadora de Fase 1 (por defecto: KMeans, K=8, UMAP 2D). Incluye ablation study (justificación de UMAP frente a PCA/sin reducción), heatmap de caracterización por clúster, análisis geográfico por departamento y validación de representatividad de la muestra de Fase 1.

## Estructura del repositorio

```
saberpro-clustering/
├── data/
│   ├── raw/            # df_maestra.csv (NO incluido, ver data/README.md)
│   ├── processed/      # Modelos y proyecciones guardadas (NO incluidos)
│   └── external/       # Diccionarios de variables, metadatos públicos del ICFES
├── notebooks/          # Notebooks guía (preprocesamiento, clustering, evaluación)
├── src/saberpro_clustering/
│   ├── preprocessing.py         # Mapeos ordinales, limpieza, escalado, one-hot
│   ├── dimensionality.py        # UMAP (con submuestreo para >100k) y PCA
│   ├── hyperparameter_search.py # Fase 1 — grid search KMeans/GMM/DBSCAN/HDBSCAN
│   ├── clustering.py            # Fase 2 — entrenamiento del modelo final
│   ├── evaluation.py            # Métricas, ablation study, validación de representatividad
│   ├── visualization.py         # Embedding UMAP 2D + heatmap de caracterización
│   ├── geo.py                   # Concentración de clústeres por departamento
│   └── config.py                # Carga de configuración (config.yaml)
├── scripts/
│   ├── run_fase1_busqueda.py       # Ejecuta la búsqueda de hiperparámetros
│   ├── run_fase2_modelo_final.py   # Entrena el modelo final y genera todos los artefactos
│   └── generate_synthetic_data.py  # Datos sintéticos para probar el pipeline sin ICFES
├── tests/               # Pruebas unitarias (pytest)
├── reports/figures/     # Figuras generadas para el artículo
├── docs/
│   └── methodology.md   # Detalle metodológico ampliado
├── config.yaml           # Hiperparámetros de Fase 1, Fase 2 y UMAP
├── environment.yml
├── requirements.txt
└── CITATION.cff
```

## Disponibilidad de los datos

Los microdatos de Saber Pro utilizados en este estudio provienen del **ICFES** y están sujetos a sus términos de uso y a la normatividad colombiana de protección de datos personales (Ley 1581 de 2012). **Este repositorio no distribuye los microdatos originales ni los datos procesados.** Ver [`data/README.md`](data/README.md) para instrucciones sobre cómo solicitar acceso a los datos oficiales y reproducir el pipeline con datos propios o sintéticos.

## Instalación

```bash
git clone https://github.com/natomo/SaberPro.git
cd SaberPro
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

O con conda:

```bash
conda env create -f environment.yml
conda activate saberpro-clustering
```

## Uso rápido

```bash
# 0. (Opcional) genera datos sintéticos para probar el pipeline sin acceso al ICFES
python scripts/generate_synthetic_data.py --n 5000 --output data/raw/df_maestra.csv

# 1. Coloca tus datos reales en data/raw/df_maestra.csv (ver data/README.md)
#    y ajusta config.yaml -> data.input_path si usas otra ruta

# 2. Fase 1 — búsqueda de hiperparámetros sobre 200k
python scripts/run_fase1_busqueda.py --config config.yaml
# Revisa reports/resultados_fase1.csv y actualiza config.yaml -> fase2
# con el algoritmo/K/dimensión UMAP ganador

# 3. Fase 2 — modelo final sobre los 452k completos
python scripts/run_fase2_modelo_final.py --config config.yaml
# Genera: modelo (.pkl), dataset con clústeres (.parquet), ablation study,
# figura de embedding UMAP, heatmap de caracterización, análisis geográfico
# y validación de representatividad — todo en reports/ y data/processed/modelos/
```

También puedes explorar el análisis paso a paso en `notebooks/`.

## Reproducibilidad

- Todas las semillas aleatorias (`random_state`) se fijan mediante `config.yaml`.
- Las versiones exactas de las librerías están congeladas en `requirements.txt`.
- Las métricas de validación de clústeres (silhouette score, índice de Davies-Bouldin, índice de Calinski-Harabasz) se calculan automáticamente y se guardan en `reports/`.

## Citación

Si utilizas este repositorio, por favor cita el artículo (ver [`CITATION.cff`](CITATION.cff), que se actualizará con los datos definitivos de publicación).

## Licencia

El código de este repositorio se distribuye bajo licencia MIT (ver [`LICENSE`](LICENSE)). Esta licencia **no aplica** a los datos de Saber Pro, cuyos derechos pertenecen al ICFES.

## Contacto

Nathalia Orozco Morales — Grupo de Investigación INTI, UNIMINUTO Pereira
