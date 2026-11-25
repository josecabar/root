import tkinter as tk
from tkinter import filedialog, messagebox
import os
import shutil
import zipfile
import tempfile
import re

import pandas as pd
import pydicom

HWL_SEQUENCE = [
    "HWL1","HWL2","HWL5","HWL6","HWL7","HWL4",
    "HWL9","HWL14","HWL17","HWL10","HWL11","HWL20",
    "HWL23","HWL12","HWL3","HWL8","HWL13","HWL15",
    "HWL18","HWL16","HWL19","HWL21","HWL24","HWL22"
]


def extract_hwl(series_description):
    """
    Extrae HWLx de Series Description.
    Ejemplos válidos:
        HWL1:G270C0T0
        HWL24
        HWL7 - Gantry90
    """

    if not series_description:
        return None

    match = re.search(
        r"(HWL\d+)",
        str(series_description),
        re.IGNORECASE
    )

    if match:
        return match.group(1).upper()

    return None


def process_input(path):

    temp_dir = None

    try:

        # ------------------------------------------------------
        # Preparar entrada
        # ------------------------------------------------------

        if path.lower().endswith(".zip"):

            temp_dir = tempfile.mkdtemp()

            with zipfile.ZipFile(path, "r") as z:
                z.extractall(temp_dir)

            input_dir = temp_dir

            base_name = os.path.splitext(
                os.path.basename(path)
            )[0]

            output_dir = os.path.join(
                os.path.dirname(path),
                base_name + "_HWL"
            )

        else:

            input_dir = path

            output_dir = os.path.join(
                os.path.dirname(path),
                os.path.basename(path) + "_HWL"
            )

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir)

        dicoms = []

        # ------------------------------------------------------
        # Leer DICOM
        # ------------------------------------------------------

        for root_dir, dirs, files in os.walk(input_dir):

            for file in files:

                if not file.lower().endswith(".dcm"):
                    continue

                filepath = os.path.join(
                    root_dir,
                    file
                )

                try:

                    ds = pydicom.dcmread(
                        filepath,
                        stop_before_pixels=True
                    )

                    acq_date = str(
                        getattr(
                            ds,
                            "AcquisitionDate",
                            ""
                        )
                    )

                    acq_time = str(
                        getattr(
                            ds,
                            "AcquisitionTime",
                            ""
                        )
                    )

                    series_description = str(
                        getattr(
                            ds,
                            "SeriesDescription",
                            ""
                        )
                    )

                    hwl = extract_hwl(
                        series_description
                    )

                    dicoms.append({
                        "path": filepath,
                        "filename": file,
                        "date": acq_date,
                        "time": acq_time,
                        "series_description": series_description,
                        "hwl": hwl
                    })

                except Exception:
                    pass

        total_dicom = len(dicoms)

        if total_dicom == 0:

            messagebox.showerror(
                "Error",
                "No se encontraron DICOM válidos."
            )

            return

        resumen = []

        # ------------------------------------------------------
        # Eliminar duplicados por fecha+hora
        # ------------------------------------------------------

        unique_by_datetime = {}

        for item in dicoms:

            key = (
                item["date"],
                item["time"]
            )

            if key not in unique_by_datetime:
                unique_by_datetime[key] = item

        unique_dicoms = list(
            unique_by_datetime.values()
        )

        n_unique = len(unique_dicoms)

        # ------------------------------------------------------
        # Identificar HWL válidos
        # ------------------------------------------------------

        valid_hwl_dicoms = [
            d for d in dicoms
            if d["hwl"] is not None
        ]

        all_have_hwl = (
            len(valid_hwl_dicoms)
            == total_dicom
        )

        # ------------------------------------------------------
        # MÉTODO A
        # Todos tienen HWL
        # y total <= 24
        # ------------------------------------------------------

        if (
            total_dicom <= 24
            and
            all_have_hwl
        ):

            exported_hwls = set()

            for item in dicoms:

                hwl = item["hwl"]

                if hwl in exported_hwls:

                    resumen.append({
                        "FileName":
                            item["filename"],
                        "AcquisitionDate":
                            item["date"],
                        "AcquisitionTime":
                            item["time"],
                        "SeriesDescription":
                            item["series_description"],
                        "Method":
                            "SeriesDescription",
                        "Status":
                            "SkippedDuplicateHWL",
                        "NewLabel":
                            hwl
                    })

                    continue

                exported_hwls.add(hwl)

                destination = os.path.join(
                    output_dir,
                    item["filename"]
                )

                shutil.copy2(
                    item["path"],
                    destination
                )

                ds = pydicom.dcmread(destination)

                ds.RTImageLabel = hwl

                ds.save_as(destination)

                resumen.append({
                    "FileName":
                        item["filename"],
                    "AcquisitionDate":
                        item["date"],
                    "AcquisitionTime":
                        item["time"],
                    "SeriesDescription":
                        item["series_description"],
                    "Method":
                        "SeriesDescription",
                    "Status":
                        "Exported",
                    "NewLabel":
                        hwl
                })

        # ------------------------------------------------------
        # MÉTODO B
        # Fallback secuencial
        # ------------------------------------------------------

        elif n_unique == 24:

            ordered_dicoms = sorted(
                unique_dicoms,
                key=lambda x: (
                    x["date"],
                    x["time"]
                )
            )

            for i, item in enumerate(
                ordered_dicoms
            ):

                new_label = HWL_SEQUENCE[i]

                destination = os.path.join(
                    output_dir,
                    item["filename"]
                )

                shutil.copy2(
                    item["path"],
                    destination
                )

                ds = pydicom.dcmread(
                    destination
                )

                ds.RTImageLabel = new_label

                ds.save_as(destination)

                resumen.append({
                    "FileName":
                        item["filename"],
                    "AcquisitionDate":
                        item["date"],
                    "AcquisitionTime":
                        item["time"],
                    "SeriesDescription":
                        item["series_description"],
                    "Method":
                        "Sequential",
                    "Status":
                        "Exported",
                    "NewLabel":
                        new_label
                })

        # ------------------------------------------------------
        # MÉTODO C
        # Menos de 24 exportables
        # ------------------------------------------------------

        elif n_unique < 24:

            exported_hwls = set()

            for item in valid_hwl_dicoms:

                hwl = item["hwl"]

                if hwl in exported_hwls:

                    resumen.append({
                        "FileName":
                            item["filename"],
                        "AcquisitionDate":
                            item["date"],
                        "AcquisitionTime":
                            item["time"],
                        "SeriesDescription":
                            item["series_description"],
                        "Method":
                            "SeriesDescription",
                        "Status":
                            "SkippedDuplicateHWL",
                        "NewLabel":
                            hwl
                    })

                    continue

                exported_hwls.add(hwl)

                destination = os.path.join(
                    output_dir,
                    item["filename"]
                )

                shutil.copy2(
                    item["path"],
                    destination
                )

                ds = pydicom.dcmread(
                    destination
                )

                ds.RTImageLabel = hwl

                ds.save_as(destination)

                resumen.append({
                    "FileName":
                        item["filename"],
                    "AcquisitionDate":
                        item["date"],
                    "AcquisitionTime":
                        item["time"],
                    "SeriesDescription":
                        item["series_description"],
                    "Method":
                        "SeriesDescription",
                    "Status":
                        "Exported",
                    "NewLabel":
                        hwl
                })

            if len(exported_hwls) == 0:

                shutil.rmtree(output_dir)

                messagebox.showerror(
                    "Error",
                    "No se encontró ningún "
                    "HWL válido."
                )

                return

        else:

            shutil.rmtree(output_dir)

            messagebox.showerror(
                "Error",
                "No se ha podido decidir "
                "un método de exportación."
            )

            return

        # ------------------------------------------------------
        # CSV resumen
        # ------------------------------------------------------

        df = pd.DataFrame(resumen)

        df.to_csv(
            os.path.join(
                output_dir,
                "resumen.csv"
            ),
            index=False
        )

        messagebox.showinfo(
            "Proceso completado",
            f"Salida creada:\n\n{output_dir}"
        )

    finally:

        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def select_folder():

    folder = filedialog.askdirectory()

    if folder:
        process_input(folder)


def select_zip():

    filename = filedialog.askopenfilename(
        filetypes=[
            ("ZIP files", "*.zip")
        ]
    )

    if filename:
        process_input(filename)


# ------------------------------------------------------
# GUI
# ------------------------------------------------------

root = tk.Tk()

root.title("HWL DICOM Renamer")
root.geometry("500x260")

title = tk.Label(
    root,
    text="HWL DICOM Renamer",
    font=("Arial", 14, "bold")
)

title.pack(pady=15)

btn_folder = tk.Button(
    root,
    text="Seleccionar carpeta",
    width=35,
    command=select_folder
)

btn_folder.pack(pady=5)

btn_zip = tk.Button(
    root,
    text="Seleccionar ZIP",
    width=35,
    command=select_zip
)

btn_zip.pack(pady=5)

info = tk.Label(
    root,
    justify="center",
    text=(
        "Método 1: Series Description (HWLx)\n"
        "Método 2: Secuencia temporal (24 imágenes)\n"
        "Genera resumen.csv con el resultado"
    )
)

info.pack(pady=20)

root.mainloop()
