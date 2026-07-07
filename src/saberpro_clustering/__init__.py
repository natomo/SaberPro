"""
saberpro_clustering
====================

Paquete con el pipeline de clustering no supervisado aplicado a los
resultados de la prueba Saber Pro (Colombia). Estrategia en dos fases:
Fase 1 busca hiperparámetros sobre una muestra de 200k; Fase 2 entrena
el modelo final sobre los 452,020 estudiantes completos.

Módulos:
    - config: carga de configuración desde config.yaml
    - preprocessing: limpieza, imputación, mapeos ordinales y codificación
    - dimensionality: reducción de dimensionalidad (UMAP, PCA)
    - hyperparameter_search: Fase 1 — grid search de KMeans/GMM/DBSCAN/HDBSCAN
    - clustering: entrenamiento del modelo final (Fase 2)
    - evaluation: métricas internas, ablation study y validación de representatividad
    - visualization: embeddings UMAP y heatmap de caracterización por clúster
    - geo: análisis de concentración de clústeres por departamento
"""

__version__ = "0.1.0"
