# Metodología detallada

> Documento complementario al artículo. Amplía las decisiones metodológicas
> que por restricciones de espacio no caben en el manuscrito.

## 1. Datos

- **Fuente:** Microdatos oficiales de la prueba Saber Pro (ICFES), año(s) [completar].
- **Tamaño muestral:** 452,020 estudiantes.
- **Unidad de análisis:** estudiante que presentó la prueba.

## 2. Preprocesamiento

1. Tratamiento de valores faltantes: [completar estrategia final usada — mediana / eliminación / imputación por grupo].
2. Codificación de variables categóricas (nivel educativo de los padres, estrato, IES, programa académico).
3. Escalamiento de variables numéricas previo a UMAP (estandarización).

## 3. Reducción de dimensionalidad

- **Algoritmo:** UMAP.
- **Hiperparámetros finales:** ver `config.yaml` → `dimensionality_reduction.umap`.
- **Justificación:** preservación de estructura local y global frente a alternativas lineales (PCA) para datos con relaciones no lineales entre variables sociodemográficas y de desempeño.

## 4. Algoritmos de clustering evaluados

| Algoritmo | Tipo                        | Ventaja principal en este estudio |
|-----------|------------------------------|-------------------------------------|
| K-Means   | Particional, basado en centroides | Rapidez, interpretabilidad de centroides |
| GMM       | Basado en modelos probabilísticos | Captura clústeres de forma elíptica/solapados |
| DBSCAN    | Basado en densidad            | Detecta ruido/outliers explícitamente |
| HDBSCAN   | Basado en densidad jerárquico | No requiere fijar número de clústeres a priori |

## 5. Validación de clústeres

- Métricas internas: silhouette score, índice de Davies-Bouldin, índice de Calinski-Harabasz.
- Análisis de estabilidad mediante remuestreo bootstrap (ver `config.yaml` → `evaluation.stability`).

## 6. Hallazgos clave a documentar

- Asimetrías en el nivel educativo de los padres identificadas entre los Clústeres 1 y 5.
- [Completar con los demás hallazgos validados sobre el parquet de datos].

## 7. Limitaciones

- [Completar: representatividad de la muestra, variables no observadas, sesgos de autoselección, etc.]
