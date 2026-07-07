# Datos

## Origen de los datos

Los microdatos de Saber Pro utilizados en este estudio son **datos abiertos** publicados por el **ICFES** (Instituto Colombiano para la Evaluación de la Educación) a través del portal de datos abiertos del Estado colombiano:

- Portal de datos abiertos: https://www.datos.gov.co
- Portal ICFES: https://www.icfes.gov.co

Al ser datos abiertos y de acceso público, **`data/raw/df_maestra.parquet` se incluye directamente en este repositorio** para garantizar la reproducibilidad completa del análisis (452,020 registros, 84 columnas).

> **Nota de formato:** el archivo se distribuye en formato **Parquet** en vez de CSV. Es exactamente el mismo dato (sin ninguna transformación), pero Parquet lo comprime de ~363 MB a ~24 MB — necesario porque GitHub bloquea archivos individuales de más de 100 MB sin Git LFS. `preprocessing.load_raw_data()` lee ambos formatos indistintamente.

## Sobre `df_maestra.parquet`

Es el dataset consolidado que resulta de descargar los resultados de Saber Pro desde el portal de datos abiertos del ICFES y unificarlos en un único archivo. Contiene, entre otras, las columnas usadas por el pipeline:

- Puntajes por competencia (`MOD_*_PUNT`)
- Variables sociodemográficas (`FAMI_*`, `ESTU_*`)
- Código de departamento de presentación (`ESTU_COD_DEPTO_PRESENTACION`)

Ver `docs/methodology.md` para el detalle de las columnas y mapeos ordinales aplicados en `preprocessing.py`.

