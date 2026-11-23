import tkinter as tk
from tkinter import filedialog, messagebox
import os
import shutil
import zipfile
import tempfile
import pandas as pd
import pydicom


HWL_SEQUENCE = [
    "HWL1","HWL2","HWL5","HWL6","HWL7","HWL4",
    "HWL9","HWL14","HWL17","HWL10","HWL11","HWL20",
    "HWL23","HWL12","HWL3","HWL8","HWL13","HWL15",
    "HWL18","HWL16","HWL19","HWL21","HWL24","HWL22"
]


def process_input(path):

    temp_dir = None

    try:

        if path.lower().endswith(".zip"):

            temp_dir = tempfile.mkdtemp()

            with zipfile.ZipFile(path, "r") as z:
                z.extractall(temp_dir)

            input_dir = temp_dir

            base_name = os.path.splitext(os.path.basename(path))[0]

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

        for root_dir, dirs, files in os.walk(input_dir):

            for file in files:

                if not file.lower().endswith(".dcm"):
                    continue

                filepath = os.path.join(root_dir, file)

                try:

                    ds = pydicom.dcmread(
                        filepath,
                        stop_before_pixels=True
                    )

                    acq_date = str(
                        getattr(ds, "AcquisitionDate", "")
                    )

                    acq_time = str(
                        getattr(ds, "AcquisitionTime", "")
                    )

                    dicoms.append({
                        "path": filepath,
                        "filename": file,
                        "date": acq_date,
                        "time": acq_time
                    })

                except Exception:
                    pass

        if len(dicoms) != 24:

            messagebox.showerror(
                "Error",
                f"Se encontraron {len(dicoms)} DICOM.\n\n"
                f"Se esperaban exactamente 24."
            )
            return

        dicoms.sort(
            key=lambda x: (
                x["date"],
                x["time"]
            )
        )

        resumen = []

        for i, item in enumerate(dicoms):

            source = item["path"]

            destination = os.path.join(
                output_dir,
                item["filename"]
            )

            shutil.copy2(source, destination)

            ds = pydicom.dcmread(destination)

            old_label = getattr(
                ds,
                "RTImageLabel",
                "SIN_ETIQUETA"
            )

            new_label = HWL_SEQUENCE[i]

            ds.RTImageLabel = new_label

            ds.save_as(destination)

            resumen.append({
                "Order": i + 1,
                "FileName": item["filename"],
                "AcquisitionDate": item["date"],
                "AcquisitionTime": item["time"],
                "OldLabel": str(old_label),
                "NewLabel": new_label
            })

        pd.DataFrame(resumen).to_csv(
            os.path.join(output_dir, "resumen.csv"),
            index=False
        )

        messagebox.showinfo(
            "Proceso completado",
            f"Archivos procesados correctamente.\n\n"
            f"Salida:\n{output_dir}"
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


root = tk.Tk()
root.title("HWL DICOM Renamer")
root.geometry("450x220")

title = tk.Label(
    root,
    text="HWL DICOM Renamer",
    font=("Arial", 14, "bold")
)
title.pack(pady=15)

btn_folder = tk.Button(
    root,
    text="Seleccionar carpeta",
    width=30,
    command=select_folder
)
btn_folder.pack(pady=5)

btn_zip = tk.Button(
    root,
    text="Seleccionar ZIP",
    width=30,
    command=select_zip
)
btn_zip.pack(pady=5)

info = tk.Label(
    root,
    text=(
        "Procesa 24 RTIMAGE DICOM,\n"
        "los ordena por Acquisition Date/Time\n"
        "y asigna las etiquetas HWL."
    ),
    justify="center"
)
info.pack(pady=15)

root.mainloop()
