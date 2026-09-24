# Código de la revisión — Saber Pro clustering (Fase 2)

Cada análisis tiene dos versiones equivalentes:
- Un **script `.py`**, pensado para un entorno propio (asume que ya corriste
  `preprocess.py` primero, que genera `X_full.npy`, `df_filtrado_full.pkl` y
  `feature_names.txt`; requiere `df_maestra.csv` en la misma carpeta o
  ajustar la ruta al inicio de cada archivo).
- Un **notebook `_colab.ipynb`**, autocontenido (carga y preprocesa
  `df_maestra.csv` desde tu Google Drive él mismo, sin depender de ningún
  otro archivo ni de haber corrido otro notebook antes) — **este es el que
  te recomiendo si solo tienes Colab**: súbelo a Colab, ajusta la ruta de
  `df_maestra.csv` en la celda de carga si hace falta, y corre todas las
  celdas en orden.

Instalar dependencias: `pip install umap-learn hdbscan scikit-learn pandas numpy scipy matplotlib --break-system-packages`

## 0. Preprocesamiento (correr primero, una sola vez)
- **preprocess.py** — replica exacta de `preprocesar_saber_pro()` de tu notebook
  original. Genera `X_full.npy` (452,020×32) y `df_filtrado_full.pkl`, que
  todos los demás scripts cargan.

## 1. Selección de K y justificación (E1 / R1-2) — Supplementary Figure S1
- Originalmente usaba directamente `resultados_fase1_2.csv` (tus resultados
  de Fase 1), sin script propio.
- **seleccion_k_y_perfiles_colab.ipynb** (nuevo) — notebook autocontenido
  para Colab que reproduce esos mismos criterios de selección de K (codo,
  Silhouette, Davies-Bouldin para K=2-8, Fase 1 n=200,000) **y además** la
  Supplementary Figure S10 (perfil estandarizado de los 8 clústeres) — las
  dos únicas figuras del suplemento que no salían de ninguno de los otros 9
  notebooks, porque pertenecen al análisis principal y no a la respuesta a
  un comentario puntual. Antes de dibujar la S10, una celda nueva
  ("4.1. Realineación de las etiquetas de clúster con la Tabla 7 publicada")
  reordena automáticamente las 8 etiquetas de esta corrida por rango de
  puntaje medio para que coincidan con la numeración ya publicada en la
  Tabla 7 (los números de K-Means son arbitrarios entre corridas
  independientes, igual que ya se advierte en la sección 10 de este
  README) — imprime el mapa de reetiquetado y la brecha mínima entre
  clústeres consecutivos, para que puedas revisar a mano si dos clústeres
  quedan demasiado cerca en puntaje. ⏱️ ~20-35 min.

## 2. Estabilidad del pipeline completo (E2 / R2-2) — Supplementary Figure S2
- **stability_analysis.py** — corre el pipeline UMAP+K-Means 15 veces,
  variando semilla de UMAP Y muestra de ajuste en cada repetición. Calcula
  ARI por pares, matriz de consenso y estabilidad por clúster.
  Salida: `stability_out/`.
- **make_stability_figure.py** — genera la figura de histograma de ARI +
  matriz de consenso a partir de `stability_out/`.
- **estabilidad_pipeline_colab.ipynb** — versión autocontenida para correr
  en tu propio Google Colab (mismo análisis).

## 3. Calibración contra datos nulos (R2-1) — Supplementary Figure S3
- **null_calibration.py** — compara el pipeline sobre datos reales vs.
  datos nulos (cada columna de X permutada de forma independiente).
  Salida: `null_calib_out/`.
- **null_calibration_strict.py** — versión metodológicamente más rigurosa:
  permuta las ETIQUETAS categóricas originales antes de codificar (evita
  combinaciones inválidas en variables one-hot). Reutiliza los resultados
  "reales" de `null_calib_out/summary.json`. Salida: `null_calib_strict_out/`.
- **calibracion_nula_colab.ipynb** — versión para correr en tu Colab.

## 4. Datos faltantes: educación de los padres (R2-4) — Supplementary Figure S5
- **missing_data_reanalysis.py** — compara 3 tratamientos de los datos
  faltantes (mediana / indicador de faltante / imputación múltiple MICE) y
  su efecto sobre las asignaciones de clúster (ARI vs. baseline).
  Salida: `missing_data_out/`.
- **datos_faltantes_colab.ipynb** — versión para Colab. ⏱️ ~20-40 min
  (corre UMAP+K-Means 9 veces: 3 variantes × 3 semillas). Ahora también
  genera la Figura S5 al final (celda nueva).

## 5. Comparación de algoritmos con espacios de búsqueda comparables (R2-3) — Supplementary Figure S4
- **algorithm_comparison_reanalysis.py** — reproduce la Fase 1 (muestra de
  200,000, semilla 42) y amplía DBSCAN a 2D-5D, HDBSCAN a min_cluster_size
  más realistas, y prueba sensibilidad de GMM a covariance_type/n_init.
  Salida: `algo_comparison_out/`.
- **comparacion_algoritmos_colab.ipynb** — versión para Colab. ⏱️ Es el más
  pesado de los 9 (barre varios eps/min_samples en 4 dimensionalidades) —
  puede tardar 45-60 min en total; guarda embeddings intermedios en Drive
  por si se desconecta. Ahora también genera la Figura S4 al final.

## 6. ¿El clúster aporta valor predictivo? (R2-6) — Supplementary Figure S6
- **predictive_value_reanalysis.py** — construye un clustering alternativo
  SOLO con covariables socioeconómicas (sin puntajes, para evitar
  circularidad) y compara Ridge/HistGBM con y sin la pertenencia al clúster
  como predictor de PUNT_GLOBAL. Salida: `predictive_value_out/`.
- **valor_predictivo_colab.ipynb** — versión para Colab. ⏱️ ~15-20 min.
  Ahora también genera la Figura S6 al final.

## 7. Análisis temporal (R2-8) — Supplementary Figure S7
- **temporal_reanalysis.py** — compara escalas de puntaje entre periodos,
  composición del clúster por periodo, y reajusta el pipeline POR SEPARADO
  en cada uno de los 4 periodos (emparejando clústeres entre periodos vía
  distancia de centroides). Salida: `temporal_out/`.
- **analisis_temporal_colab.ipynb** — versión para Colab. ⏱️ ~15-25 min
  (un UMAP fit por periodo, 4 en total). Ahora también genera la Figura S7
  al final.

## 8. Sensibilidad institucional / de programa (R2-10) — Supplementary Figure S8
- **institutional_sensitivity_reanalysis.py** — reporta concentración
  institucional/de programa por clúster (HHI) y hace un análisis
  leave-institution-out / leave-program-out sobre las 5 instituciones y 5
  programas más grandes. Salida: `institutional_out/`.
- **sensibilidad_institucional_colab.ipynb** — versión para Colab. ⏱️
  ~20-30 min (11 ajustes de UMAP: 1 base + 10 de leave-out). Ahora también
  genera la Figura S8 al final.

## 9. Caracterización del clúster pequeño (R2-11) — Supplementary Figure S9
- **cluster7_reanalysis.py** — localiza el clúster pequeño (~1-1.5%),
  intenta separar sus "dos islas" (DBSCAN / KMeans de respaldo), y
  caracteriza cada subgrupo (puntaje, instituciones, internet, etc.).
  Salida: `cluster7_out/`.
- **cluster_pequeno_colab.ipynb** — versión para Colab. ⏱️ ~10-15 min. La
  reproducibilidad de este clúster a través de 15 corridas se calcula por
  separado en `estabilidad_pipeline_colab.ipynb` (punto 2). Ahora también
  genera la Figura S9 al final.

## 10. Tamaños de efecto y cobertura muestral (R2-5 / R2-9)
- **effect_sizes_and_coverage.py** (paso 1) — reajusta el pipeline
  completo (UMAP fit en submuestra de 80,000, semilla 42, transform sobre
  las 452,020 filas; MiniBatchKMeans K=8) y guarda las etiquetas y
  PUNT_GLOBAL por estudiante en `effect_size_out/labels_and_scores.csv`,
  además de comparar tamaños de clúster contra la Tabla 7 real (por rango
  de puntaje medio, no por número de etiqueta — las etiquetas de K-Means
  son arbitrarias entre corridas).
- **effect_sizes_step2.py** (paso 2, correr después del paso 1) — con
  `labels_and_scores.csv`, calcula eta-cuadrado (ANOVA de un factor sobre
  PUNT_GLOBAL por clúster) y d de Cohen por pares entre los 8 clústeres
  (28 pares), emparejando por rango con la Tabla 7. Reproduce los números
  reportados en el manuscrito: eta² = 0.195, d entre 0.05 y 3.53
  (mediana = 0.59), tamaños de clúster dentro de 0.1-9.5% de la Tabla 7.
  R2-9 (cobertura muestral) no requiere reanálisis: se resolvió con
  `df['ESTU_CONSECUTIVO'].value_counts()` directamente sobre
  `df_filtrado_full.pkl` (1,051 de 452,020 registros comparten
  identificador → 450,969 identificadores únicos).
- **tamanos_efecto_cobertura_colab.ipynb** — versión para Colab (pasos 1 y
  2 combinados en un solo notebook). ⏱️ ~10-20 min.

## Carta de respuesta a revisores
- **build_response_letter.js** (Node.js, usa el paquete `docx`) — genera
  `Respuesta_a_revisores_borrador.docx` completo a partir de todo el texto
  redactado. Correr con `node build_response_letter.js`.

## Plan de revisión (Excel)
- **build_plan.py** / **update_plan_full_text.py** — construyeron y
  poblaron inicialmente `plan_revision_saberpro.xlsx` (las 19 filas con
  comentarios traducidos). Las actualizaciones posteriores por punto se
  hicieron con scripts puntuales más pequeños (no incluidos aquí, eran
  ediciones de una sola vez); si quieres el historial completo de esos,
  dime y te los paso también.

## Orden recomendado para volver a correr todo desde cero
1. `preprocess.py`
2. `stability_analysis.py` (necesario para varios de los siguientes, ya
   que reutilizan su `idx_eval.npy` y `labels_all.npy`)
3. `null_calibration.py` → `null_calibration_strict.py`
4. `missing_data_reanalysis.py`
5. `algorithm_comparison_reanalysis.py`
6. `predictive_value_reanalysis.py`
7. `temporal_reanalysis.py`
8. `institutional_sensitivity_reanalysis.py`
9. `cluster7_reanalysis.py`
10. `effect_sizes_and_coverage.py` → `effect_sizes_step2.py`
11. `node build_response_letter.js`

## Nota para correr en Google Colab
Si solo tienes Colab, usa directamente los 10 archivos `_colab.ipynb` — no
necesitas los `.py` para nada, cada notebook es independiente y no depende
de haber corrido otro antes (cada uno reconstruye su propia partición de
referencia K=8 en vez de reutilizar la de otro notebook). Los 9 notebooks
de reanálisis ahora generan también su propia figura suplementaria al
final (S2-S9); el décimo (`seleccion_k_y_perfiles_colab.ipynb`) genera las
dos figuras que no pertenecen a ningún comentario puntual de revisor (S1 y
S10). Para cada uno:

1. Sube `df_maestra.csv` a tu Google Drive, en `MyDrive/Proyecto/`
   (o ajusta la ruta en la celda "Cargar los datos desde Google Drive").
2. Abre el `.ipynb` en Colab (`Archivo` → `Subir notebook`, o arrástralo a
   la ventana de Colab).
3. `Entorno de ejecución` → `Cambiar tipo de entorno` → **CPU** (ninguno
   de estos 10 notebooks necesita GPU).
4. `Entorno de ejecución` → `Ejecutar todas`. La primera celda te pedirá
   autorizar el acceso a tu Drive — acéptalo, es necesario para leer
   `df_maestra.csv` y guardar los resultados ahí mismo.
5. Los resultados quedan en tu Drive, en `MyDrive/Proyecto/<nombre_del_análisis>/`,
   guardados progresivamente (por si la sesión se desconecta a mitad de camino).

Los tiempos aproximados de cada uno están anotados junto a su entrada en
las secciones 1-10 de arriba; en total, correr los 10 de punta a punta en
Colab gratuito puede tomar entre 3.5 y 4.5 horas — no hace falta correrlos
todos de una sentada, cada uno es independiente.

Si prefieres los `.py` en algún otro entorno propio (no Colab), también
corren ahí: súbelos junto con `df_maestra.csv`, ajusta las rutas al inicio
de cada script (por defecto usan `/home/claude/...`), instala las
dependencias de la primera línea de este README, y corre cada uno con
`!python nombre_del_script.py` (o `python nombre_del_script.py` en una
terminal normal).
