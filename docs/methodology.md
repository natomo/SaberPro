# Metodología detallada

## 1. Datos

- **Fuente:** Microdatos oficiales de la prueba Saber Pro (ICFES), año(s) 2021-2 a 2023-1.
- **Tamaño muestral:** 452,020 estudiantes.
- **Unidad de análisis:** estudiante que presentó la prueba.

## 2. Preprocesamiento

1. Tratamiento de valores faltantes: **imputación por tipo de variable** (no un único método global), aplicada en `preprocessing.preprocesar_saber_pro` antes de escalar/codificar:
   - **Puntajes** (`MOD_*_PUNT`, continuas): imputación con la **media** de cada columna.
   - **Variables ordinales** (estrato, valor matrícula, educación de los padres, horas de trabajo): imputación con la **mediana**, tras aplicar los mapeos ordinales (ver nota sobre cobertura de mapeos más abajo).
   - **Variables nominales/binarias** (beca, crédito, internet, computador, etc.): imputación con la **moda** (valor más frecuente).
   - Filas sin ningún puntaje registrado se eliminan directamente (`dropna(subset=columnas_puntaje)`), en vez de imputarse — un estudiante sin ningún resultado no aporta información válida para el clustering.
   - Cualquier NaN residual que sobreviva a lo anterior (por ejemplo, por columnas no cubiertas por los pasos previos) se imputa como respaldo con la mediana sobre la matriz `X` ya construida.
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

- **Asimetría en educación parental** (Clústeres 1 y 5): un clúster con alta educación paterna y baja materna, y su patrón inverso — la asimetría entre padres, no solo el nivel educativo combinado, emerge como eje estructural distintivo.
- **Exclusión digital severa en territorios de frontera** (Clúster 6): z-scores fuertemente negativos en internet, computador y activos del hogar, concentrado geográficamente en los departamentos más aislados (Vaupés, Chocó, Putumayo, La Guajira).
- **Vulnerabilidad estructural por presión económica** (Clúster 0): alta intensidad de horas de trabajo semanal y financiamiento con recursos propios, en Casanare, Guainía y Guaviare.
- **Élite académica minoritaria** (Clúster 7, ~1.5% de la muestra): rendimiento excepcional y consistente en los cinco módulos, muy por encima del resto de clústeres.
- **Validación geográfica externa**: aunque el departamento de presentación se excluyó de la matriz de features, los clústeres muestran alineación espontánea con patrones regionales conocidos (gradiente frontera-periferia, concentración en costa Atlántica, ventaja urbana en Bogotá/Valle del Cauca) — evidencia de que los perfiles capturan estructura socioeconómica real y no artefactos del algoritmo.
- **Necesidad metodológica de UMAP**: el ablation study confirma que la reducción no lineal captura estructura de manifold no recuperable con PCA ni en el espacio original de 32 dimensiones.
- **Estabilidad del modelo**: K=8 se mantiene consistente a través de 10 semillas aleatorias (Silhouette CV=2.3%), con variabilidad de asignación de etiquetas (ARI=0.688) atribuible a inicialización de centroides más que a inestabilidad estructural.
- **Convergencia multi-métrica**: K=8 es la única configuración donde Silhouette, Calinski-Harabasz y Davies-Bouldin convergen simultáneamente en valores óptimos, frente a alternativas con mayor fragmentación (HDBSCAN) o exceso de ruido (DBSCAN).

## 7. Limitaciones

- **Representatividad de la muestra de Fase 1**: aunque se validó formalmente (diferencias de medias <0.25%, p-valores KS >0.17 en las 10 variables continuas/ordinales), la búsqueda de hiperparámetros se hizo sobre 200,000 de los 452,020 registros; existe un margen teórico de que configuraciones óptimas para el subconjunto no sean exactamente óptimas para la población completa.
- **Variables no observadas**: el estudio no incluye ingreso familiar directo, capital cultural, o características institucionales de la IES de origen — solo proxies socioeconómicos (estrato, valor de matrícula, tenencia de activos).
- **Sesgo de autoselección**: la muestra incluye únicamente estudiantes que efectivamente presentaron Saber Pro; estudiantes que abandonaron sus estudios antes de este punto no están representados, lo que puede subestimar la magnitud real de las desigualdades estructurales.
- **Exclusión de cohortes COVID-19** (2020-1 y 2020-2): necesaria para evitar confusión con condiciones de educación remota de emergencia, pero limita la generalización de los hallazgos a periodos de operación educativa regular.
- **Supuestos geométricos de K-Means**: asume clústeres aproximadamente esféricos en el espacio UMAP; aunque el ablation study y la comparación con GMM/DBSCAN/HDBSCAN mitigan esta preocupación, no se puede descartar que estructuras no convexas queden subrepresentadas.
- **Etiquetado de perfiles post-hoc**: los nombres de los clústeres (ej. "élite académica", "exclusión digital") son interpretación de los autores basada en z-scores y validación geográfica, no una salida directa del algoritmo — otros investigadores podrían priorizar variables distintas al caracterizar los mismos ocho grupos.
- **Alcance geográfico único**: los hallazgos corresponden a la estructura socioeconómica y educativa de Colombia; la transferibilidad a otros sistemas de educación superior con distinta composición institucional o política de acceso no está garantizada.
