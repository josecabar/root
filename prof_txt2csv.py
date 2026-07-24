#!/usr/bin/env python3

import os
from tkinter import Tk
from tkinter.filedialog import askdirectory
from tkinter.filedialog import askopenfilename

from pathlib import Path
import re

def generar_csv_haz(
    lines,
    haz_idx,
    prm_name,
    txt_file
):

    csv_name = os.path.join(
        os.path.dirname(txt_file),
        Path(prm_name).stem + ".csv"
    )

    output = []

    start = next(
        i for i, l in enumerate(lines)
        if l.strip().startswith(
            "X Axis Analysis"
        )
    )

    end = next(
        i for i, l in enumerate(lines)
        if l.strip().startswith(
            "Measured Data:"
        )
    )

    for line in lines[start:end]:

        line = line.strip()

        if not line:
            continue

        if line.endswith("Analysis"):

            output.append(
                line + ",,,"
            )
            continue

        parts = [
            p.strip()
            for p in line.split("\t")
        ]

        print(parts)

        parts = [
            p for p in parts
            if p != ""
        ]

        if len(parts) < 3:
            continue

        nombre = parts[0]
        unidad = parts[1]

        columna = haz_idx + 3

        if columna >= len(parts):
            continue

        valor = parts[columna]

        try:
            valor = float(
                valor.replace(",", ".")
            )

            valor = f"{valor:.3f}"

        except ValueError:
            pass
        
        if (
            nombre.startswith("Penumbra")
            and unidad == "cm"
        ):
            try:
                valor = (
                    float(
                        valor.replace(",", ".")
                    ) * 10
                )
                valor = f"{valor:.4f}"
                unidad = "mm"
            except:
                pass

        output.append(
            f"{nombre},{unidad},{valor}"
        )

    with open(
        csv_name,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(output)
        )

    print(
        f"[OK] {csv_name}"
    )

def obtener_prm_names(lines):

    filename_line = None

    for linea in lines:

        if "Filename" in linea:

            filename_line = linea
            break

    if filename_line is None:

        return []

    prm_names = []

    rutas = re.findall(
        r'[^\\/\t]+\.(?:prm|PRM)',
        filename_line
    )

    for ruta in rutas:

        prm_names.append(
            Path(ruta).name
        )

    print("\nPRM encontrados:")
    print(prm_names)

    return prm_names

def profiler_txt_to_csv(txt_file):

    with open(
        txt_file,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        lines = f.readlines()

    prm_names = obtener_prm_names(lines)

    if not prm_names:

        raise ValueError(
            "No se encontraron archivos PRM en la línea Filename"
        )

    output_files = []

    # TXT individual
    if len(prm_names) == 1:

        generar_csv_haz(
            lines,
            0,
            prm_names[0],
            txt_file
        )

        output_files.append(
            str(
                Path(
                    prm_names[0]
                ).with_suffix(".csv")
            )
        )

    # TXT multihaz
    else:

        for haz_idx, prm_name in enumerate(
            prm_names
        ):

            generar_csv_haz(
                lines,
                haz_idx,
                prm_name,
                txt_file
            )

            output_files.append(
                str(
                    Path(
                        prm_name
                    ).with_suffix(".csv")
                )
            )

    return output_files

if __name__ == "__main__":

    root = Tk()
    root.withdraw()

    txt_file = askopenfilename(
        title="Selecciona TXT exportado por IC Profiler",
        filetypes=[
            ("TXT", "*.txt"),
            ("Todos", "*.*")
        ]
    )

    root.destroy()

if txt_file:

    archivos = profiler_txt_to_csv(
        txt_file
    )

    print(
        "\nCSV generados:"
    )

    for f in archivos:

        print(f)
