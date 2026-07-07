#!/usr/bin/env python
"""Genera un dataset sintético con la estructura real de df_maestra.csv
(columnas y categorías de Saber Pro) para poder probar el pipeline sin
acceso a los microdatos oficiales del ICFES.

No representa datos reales de ningún estudiante.

Uso:
    python scripts/generate_synthetic_data.py --n 2000 --output data/raw/df_maestra.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    estratos = ["Sin estrato", "Estrato 1", "Estrato 2", "Estrato 3", "Estrato 4", "Estrato 5", "Estrato 6"]
    educacion = [
        "Ninguno", "Primaria incompleta", "Primaria completa",
        "Secundaria (Bachillerato) incompleta", "Secundaria (Bachillerato) completa",
        "Técnica o tecnológica incompleta", "Técnica o tecnológica completa",
        "Educación profesional incompleta", "EDUCACIÓN PROFESIONAL COMPLETA", "POSTGRADO",
    ]
    matricula = [
        "Sin costo", "Menos de 500 mil", "Entre 500 mil y menos de 1 millón",
        "Entre 1 millón y menos de 2.5 millones", "Entre 2.5 millones y menos de 4 millones",
        "Entre 4 millones y menos de 5.5 millones", "Entre 5.5 millones y menos de 7 millones",
        "Más de 7 millones",
    ]
    horas_trabajo = ["0", "Menos de 10 horas", "Entre 11 y 20 horas", "Entre 21 y 30 horas", "Más de 30 horas"]
    si_no = ["Si", "No"]
    deptos_codigos = [5, 8, 11, 13, 15, 17, 66, 68, 76]

    df = pd.DataFrame(
        {
            "MOD_RAZONA_CUANTITAT_PUNT": rng.normal(150, 25, n).clip(0, 300).round(1),
            "MOD_LECTURA_CRITICA_PUNT": rng.normal(150, 25, n).clip(0, 300).round(1),
            "MOD_COMPETEN_CIUDADA_PUNT": rng.normal(150, 25, n).clip(0, 300).round(1),
            "MOD_INGLES_PUNT": rng.normal(150, 25, n).clip(0, 300).round(1),
            "MOD_COMUNI_ESCRITA_PUNT": rng.normal(150, 25, n).clip(0, 300).round(1),
            "FAMI_ESTRATOVIVIENDA": rng.choice(estratos, n, p=[0.03, 0.15, 0.35, 0.28, 0.12, 0.05, 0.02]),
            "ESTU_VALORMATRICULAUNIVERSIDAD": rng.choice(matricula, n),
            "FAMI_EDUCACIONPADRE": rng.choice(educacion, n),
            "FAMI_EDUCACIONMADRE": rng.choice(educacion, n),
            "ESTU_HORASSEMANATRABAJA": rng.choice(horas_trabajo, n),
            "ESTU_TITULOOBTENIDOBACHILLER": rng.choice(["Bachiller académico", "Bachiller técnico"], n),
            "ESTU_PAGOMATRICULABECA": rng.choice(si_no, n, p=[0.2, 0.8]),
            "ESTU_PAGOMATRICULACREDITO": rng.choice(si_no, n, p=[0.3, 0.7]),
            "ESTU_PAGOMATRICULAPADRES": rng.choice(si_no, n, p=[0.5, 0.5]),
            "ESTU_PAGOMATRICULAPROPIO": rng.choice(si_no, n, p=[0.25, 0.75]),
            "ESTU_COMOCAPACITOEXAMENSB11": rng.choice(["Por internet", "Con un curso", "Con libros"], n),
            "FAMI_TIENEINTERNET": rng.choice(si_no, n, p=[0.7, 0.3]),
            "FAMI_TIENECOMPUTADOR": rng.choice(si_no, n, p=[0.65, 0.35]),
            "FAMI_TIENEAUTOMOVIL": rng.choice(si_no, n, p=[0.3, 0.7]),
            "FAMI_TIENELAVADORA": rng.choice(si_no, n, p=[0.6, 0.4]),
            "ESTU_COD_DEPTO_PRESENTACION": rng.choice(deptos_codigos, n),
        }
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera datos sintéticos tipo Saber Pro (df_maestra)")
    parser.add_argument("--n", type=int, default=2000, help="Número de registros a generar")
    parser.add_argument("--output", default="data/raw/df_maestra.csv", help="Ruta de salida (.csv o .parquet)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate(args.n, args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    print(f"Dataset sintético con {args.n:,} registros guardado en {output_path}")


if __name__ == "__main__":
    main()
