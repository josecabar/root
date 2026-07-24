#!/usr/bin/env python3

import os
from tkinter import Tk
from tkinter.filedialog import askdirectory

def profiler_txt_to_csv(txt_file):

    csv_file = os.path.splitext(txt_file)[0] + ".csv"

    with open(
        txt_file,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        lines = f.readlines()

    # --------------------------------------
    # Buscar inicio
    # --------------------------------------

    start = None

    for i, line in enumerate(lines):

        if line.strip() == "X Axis Analysis":

            start = i
            break

    if start is None:

        raise ValueError(
            "No se encontró 'X Axis Analysis'"
        )

    # --------------------------------------
    # Buscar final
    # --------------------------------------

    end = len(lines)

    for i, line in enumerate(lines):

        if line.startswith("Measured Data:"):

            end = i
            break

    lines = lines[start:end]

    output = []

    # --------------------------------------
    # Procesar
    # --------------------------------------

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Secciones
        if line.endswith("Analysis"):

            output.append(
                f"{line},,"
            )

            continue

        parts = [
            p.strip()
            for p in line.split("\t")
            if p.strip()
        ]

        if len(parts) < 3:
            continue

        parametro = parts[0]
        unidad = parts[1]
        valor = parts[2]

        # Penumbra cm -> mm
        if (
            parametro.startswith("Penumbra")
            and unidad == "cm"
        ):

            try:

                valor = (
                    float(valor) * 10
                )

                valor = f"{valor:.4f}"

                unidad = "mm"

            except ValueError:

                pass

        output.append(
            f"{parametro},{unidad},{valor}"
        )

    # --------------------------------------
    # Guardar CSV
    # --------------------------------------

    with open(
        csv_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(output)
        )

    return csv_file



if __name__ == "__main__":

    root = Tk()
    root.withdraw()

    folder = askdirectory(
        title="Selecciona carpeta con archivos TXT del Profiler"
    )

    root.destroy()

    if folder:

        txt_files = [
            f for f in os.listdir(folder)
            if f.lower().endswith(".txt")
        ]

        for file_name in txt_files:

            txt_file = os.path.join(
                folder,
                file_name
            )

            try:

                csv_file = profiler_txt_to_csv(
                    txt_file
                )

                print(
                    f"[OK] {file_name} -> "
                    f"{os.path.basename(csv_file)}"
                )

            except Exception as e:

                print(
                    f"[ERROR] {file_name}: {e}"
                )

        print(
            "\nConversión finalizada."
        )
