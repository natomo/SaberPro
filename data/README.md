# Datos

## Por qué no hay datos en este repositorio

Los microdatos de Saber Pro son propiedad del **ICFES** (Instituto Colombiano para la Evaluación de la Educación) y su distribución está sujeta a:

- Los términos y condiciones de uso del ICFES para investigadores.
- La Ley 1581 de 2012 (protección de datos personales) y normas concordantes en Colombia.
- Los acuerdos institucionales entre UNIMINUTO y el ICFES bajo los cuales se obtuvo el dataset de 452,020 estudiantes.

Por estas razones, **`data/raw/` y `data/processed/` se mantienen vacías** (solo con archivos `.gitkeep`) y están excluidas explícitamente en `.gitignore`.

## Cómo obtener acceso a los datos oficiales

1. Consulta el portal de datos abiertos del ICFES: https://www.icfes.gov.co
2. Revisa los términos de uso para investigación académica.
3. Si tu institución tiene un convenio vigente (como el de UNIMINUTO), solicita el dataset a través del canal institucional correspondiente.

## Formato esperado para reproducir el pipeline

Si cuentas con acceso a datos equivalentes (propios o de una convocatoria distinta), coloca un archivo en `data/raw/saber_pro.parquet` (o `.csv`) con, como mínimo, las siguientes columnas (los nombres exactos se configuran en `config.yaml`):

| Columna                          | Tipo    | Descripción                                      |
|-----------------------------------|---------|---------------------------------------------------|
| id_estudiante                     | string  | Identificador anonimizado del estudiante           |
| puntaje_global                     | float   | Puntaje global Saber Pro                           |
| puntajes_competencias_*            | float   | Puntajes por competencia genérica                  |
| educacion_padre / educacion_madre | category| Nivel educativo de los padres                      |
| estrato, ies, programa, ...        | category| Variables sociodemográficas y académicas adicionales|

## Datos sintéticos para pruebas

Para desarrollo y pruebas unitarias sin datos reales, usa:

```bash
python scripts/generate_synthetic_data.py --n 2000 --output data/raw/synthetic_sample.parquet
```

Esto genera un dataset sintético con la misma estructura de columnas, útil para validar que el pipeline corre correctamente antes de ejecutarlo sobre datos reales.
