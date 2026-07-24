import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

from tkinter import Tk, simpledialog
from tkinter.filedialog import askopenfilename
from tkinter import filedialog, messagebox

import os

import re

def obtener_campo(nombre_archivo):

    match = re.search(
        r'(\d+)\s*x\s*(\d+)',
        nombre_archivo,
        re.IGNORECASE
    )

    if not match:
        raise ValueError(
            "No se pudo determinar el tamaño de campo"
        )

    campo = (
        f"{match.group(1)}x{match.group(2)}"
    )

    return campo

def guardar_resultados_csv(resultados):
    """
    resultados = {
        "Campo": "10x10",
        "Unflatness": 1.312,
        "SlopeAvg": 0.487,
        "Symmetry": 100.21,
        "PeakPosition": 0.11,
        "GaussianOffset": 0.24
    }
    """

    guardar = messagebox.askyesno(
        "Guardar resultados",
        "¿Desea guardar los resultados en un archivo CSV?"
    )

    if not guardar:
        return

    csv_path = filedialog.asksaveasfilename(
        title="Guardar/Seleccionar CSV",
        defaultextension=".csv",
        filetypes=[("CSV", "*.csv")]
    )

    if not csv_path:
        return

    energia = "6MVFFF"

    columnas = [
        "Energia",
        "Campo",
        "Unflatness",
        "SlopeAvg",
        "Symmetry",
        "PeakPosition",
        "GaussianOffset"
    ]

    orden_campos = [
        "7x7",
        "10x10",
        "20x20"
    ]

    # --------------------------------------------------
    # Crear CSV nuevo
    # --------------------------------------------------

    if not os.path.exists(csv_path):

        df = pd.DataFrame({
            "Energia": [energia] * 3,
            "Campo": ["7x7", "10x10", "20x20"],

            "X_FieldSize": [np.nan] * 3,
            "X_PenumbraLeft": [np.nan] * 3,
            "X_PenumbraRight": [np.nan] * 3,
            "X_Unflatness": [np.nan] * 3,
            "X_SlopeAvg": [np.nan] * 3,
            "X_Symmetry": [np.nan] * 3,
            "X_PeakPosition": [np.nan] * 3,
            "X_GaussianOffset": [np.nan] * 3,

            "Y_FieldSize": [np.nan] * 3,
            "Y_PenumbraBottom": [np.nan] * 3,
            "Y_PenumbraTop": [np.nan] * 3,
            "Y_Unflatness": [np.nan] * 3,
            "Y_SlopeAvg": [np.nan] * 3,
            "Y_Symmetry": [np.nan] * 3,
            "Y_PeakPosition": [np.nan] * 3,
            "Y_GaussianOffset": [np.nan] * 3
        })

    else:

        df = pd.read_csv(
            csv_path
        )

        # Garantizar existencia de filas

        campo_actual = resultados["Campo"]

        mask = (
            (df["Energia"] == energia)
            &
            (df["Campo"] == campo_actual)
        )

            
        for clave, valor in resultados.items():

            if clave in ["Campo"]:
                continue

            df.loc[mask, clave] = valor

    # --------------------------------------------------
    # Actualizar fila correspondiente
    # --------------------------------------------------

    campo_actual = resultados["Campo"]

    mask = (
        (df["Energia"] == energia)
        &
        (df["Campo"] == campo_actual)
    )

    for clave, valor in resultados.items():

        if clave == "Campo":
            continue

        df.loc[mask, clave] = float(valor)

    # --------------------------------------------------
    # Orden fijo 7x7, 10x10, 20x20
    # --------------------------------------------------

    orden = {
        "7x7": 1,
        "10x10": 2,
        "20x20": 3
    }

    df["__orden"] = (
        df["Campo"]
        .map(orden)
    )

    df = (
        df.sort_values("__orden")
        .drop(columns="__orden")
    )

    df.to_csv(
        csv_path,
        index=False
    )

    messagebox.showinfo(
        "CSV guardado",
        f"Resultados guardados en:\n{csv_path}"
    )


def process_file(file_path):

    resultados_export = {}


    nombre_archivo = os.path.basename(
        file_path
    )

    net_doses = load_ic_profiler_prm(
        file_path
    )

    field_size_cm = (
        get_field_size_from_filename(
            nombre_archivo
        )
    )

    print("\n" + "=" * 60)
    print("ESTUDIO FOGLIATA")
    print("=" * 60)

    for eje in ["X", "Y", "PD", "ND"]:

        data = analyze_fogliata_linear(
            net_doses,
            eje,
            field_size_cm,
            depth_cm=3.9
        )

        res = data["results"]

        if eje == "X":

            resultados_export["X"] = {
                **res,
                "field_size": data["field_size"],
                "pen_left": data["pen_left"],
                "pen_right": data["pen_right"]
            }

        if eje == "Y":

            resultados_export["Y"] = {
                **res,
                "field_size": data["field_size"],
                "pen_left": data["pen_left"],
                "pen_right": data["pen_right"]
            }

        print(f"\nEJE {eje}")

        print(
            f"Renorm Factor : {res['RenormFactor']:.2f}"
        )

        print(
            f"Unflatness    : {res['Unflatness']:.3f}"
        )

        print(
            f"Slope Avg     : {res['SlopeAvg']:.4f}"
        )

        print(
            f"Symmetry      : {res['Symmetry']:.2f} %"
        )

        print(
            f"Field Size    : {data['field_size']:.2f} mm"
        )

        print(
            f"Field Region  : {data['field_region']:.2f} mm"
        )

        print(
            f"Left Penumbra : {data['pen_left']:.2f} mm"
        )

        print(
            f"Right Penumbra: {data['pen_right']:.2f} mm"
        )

    print("\nAnalisis completado.")


    # ==========================================
    # GRAFICA 2x2
    # ==========================================

    ejes = ["X", "Y", "PD", "ND"]

    fig, axs = plt.subplots(
        2,
        2,
        figsize=(16, 10)
    )

    axes_map = {
        "X": axs[0, 0],
        "Y": axs[0, 1],
        "PD": axs[1, 0],
        "ND": axs[1, 1]
    }

    for eje in ejes:

        data = analyze_fogliata_linear(
            net_doses,
            eje,
            field_size_cm,
            depth_cm=3.9
        )

        res = data["results"]

        ax = axes_map[eje]

        x = data["x"]
        y = data["y"]
        f = data["f"]

        # =========================
        # Datos
        # =========================

        l20 = data["l20"]
        l50 = data["l50"]
        l80 = data["l80"]

        r20 = data["r20"]
        r50 = data["r50"]
        r80 = data["r80"]

        fr_left = data["fr_left"]
        fr_right = data["fr_right"]

        field_size = data["field_size"]
        field_region = data["field_region"]

        pen_left = data["pen_left"]
        pen_right = data["pen_right"]

        x1_l = data["x1_l"]
        x2_l = data["x2_l"]

        x1_r = data["x1_r"]
        x2_r = data["x2_r"]

        D1_l = data["D1_l"]
        D2_l = data["D2_l"]

        D1_r = data["D1_r"]
        D2_r = data["D2_r"]

        # =========================
        # Perfil completo
        # =========================

        x_plot = np.linspace(
            np.min(x),
            np.max(x),
            5000
        )

        y_plot = f(x_plot)

        # Curva completa naranja

        ax.plot(
            x_plot,
            y_plot,
            color="darkorange",
            lw=1.5
        )

        ax.scatter(
            x,
            y,
            color="black",
            s=8,
            alpha=0.6,
            zorder=5
        )

        # Región de campo (verde)

        mask_fr = (
            (x_plot >= fr_left)
            &
            (x_plot <= fr_right)
        )

        ax.plot(
            x_plot[mask_fr],
            y_plot[mask_fr],
            color="forestgreen",
            lw=1.5
        )
        
        # =========================
        # Penumbra (20%-80%)
        # =========================

        for v in [l20, l80, r80, r20]:

            ax.plot(
                [v, v],
                [0, 100],
                color="royalblue",
                ls="--",
                lw=1.5
            )

        # =========================
        # Field Region
        # =========================

        y_top = np.max(y)

        ax.plot(
            [fr_left, fr_left],
            [80, y_top],
            color="green",
            ls="--",
            lw=1.5
        )

        ax.plot(
            [fr_right, fr_right],
            [80, y_top],
            color="green",
            ls="--",
            lw=1.5
        )

        # =========================
        # F. Size
        # =========================

        ax.annotate(
            "",
            xy=(l50, 50),
            xytext=(r50, 50),
            arrowprops=dict(
                arrowstyle="<->",
                color="black",
                lw=1.5
            )
        )

        ax.text(
            0,
            53,
            f"F. Size {field_size:.2f} mm",
            ha="center"
        )

        # =========================
        # F. Region
        # =========================

        ax.annotate(
            "",
            xy=(fr_left, 90),
            xytext=(fr_right, 90),
            arrowprops=dict(
                arrowstyle="<->",
                color="black",
                lw=1.5
            )
        )

        ax.text(
            0,
            93,
            f"F. Region {field_region:.2f} mm",
            ha="center"
        )

        # =========================
        # Slope points
        # =========================

        ax.scatter(
            [x1_l, x2_l, x1_r, x2_r],
            [D1_l, D2_l, D1_r, D2_r],
            color="royalblue",
            s=40,
            zorder=2
        )

        ax.plot(
            [x1_l, x2_l],
            [D1_l, D2_l],
            color="red",
            linestyle="--",
            lw=1.5
        )

        ax.plot(
            [x1_r, x2_r],
            [D1_r, D2_r],
            color="red",
            linestyle="--",
            lw=1.5
        )

        # =========================
        # Gaussian plot
        # =========================

        A_fit = data["A_fit"]
        mu_fit = data["Mu_fit"]
        sigma_fit = data["Sigma_fit"]
        C_fit = data["C_fit"]

        mask_gauss = (
            (x_plot >= l20)
            &
            (x_plot <= r20)
        )

        y_gauss = gaussian(
            x_plot,
            A_fit,
            mu_fit,
            sigma_fit,
            C_fit
        )

        ax.plot(
            x_plot[mask_gauss],
            y_gauss[mask_gauss],
            color="#b0b7ff",
            linestyle=":",
            linewidth=1.5,
            alpha=0.9
        )



        # =========================
        # Cuadro izquierdo
        # =========================

        txt_left = (
            f"Gaussian Offset = {res['GaussianOffset']:.2f} mm\n"
            f"Left Penumbra = {pen_left:.2f} mm\n"
            f"Left Slope = {res['SlopeLeft']:.4f}\n"
            f"Slope Avg = {res['SlopeAvg']:.4f}\n"
            f"Unflatness = {res['Unflatness']:.3f}"
        )

        ax.text(
            0.02,
            0.98,
            txt_left,
            transform=ax.transAxes,
            va="top",
            zorder=50,
            bbox=dict(
                facecolor="wheat",
                alpha=0.90
            )
        )

        # =========================
        # Cuadro derecho
        # =========================

        txt_right = (
            f"Renorm Factor = {res['RenormFactor']:.2f}\n"
            f"Right Penumbra = {pen_right:.2f} mm\n"
            f"Right Slope = {res['SlopeRight']:.4f}\n"
            f"Symmetry = {res['Symmetry']:.2f} %\n"
            f"Peak Position = {res['PeakPosition']:.2f} mm"
        )
        ax.text(
            0.68,
            0.98,
            txt_right,
            transform=ax.transAxes,
            va="top",
            zorder=50,
            bbox=dict(
                facecolor="wheat",
                alpha=0.90
            )
        )

        # =========================
        # Formato
        # =========================

        ax.set_title(
            f"Eje {eje}"
        )

        ax.set_xlabel(
            "Off-axis (mm)"
        )

        ax.set_ylabel(
            "Dose (%)"
        )

        ax.grid(
            True,
            alpha=0.25
        )

        ax.set_xlim(
            np.min(x),
            np.max(x)
        )


    # ==========================================
    # Título general
    # ==========================================

    fig.suptitle(
        f"{nombre_archivo}\n"
        f"IC Profiler - Elekta Versa HD - 6 MV FFF",
        fontsize=16
    )
    plt.tight_layout()

    plt.show()

    return resultados_export

    
def gaussian(x, A, mu, sigma, C):
    return (
        A * np.exp(
            -((x - mu)**2) /
            (2 * sigma**2)
        )
        + C
    )

def get_field_size_from_filename(filename):

    import re
    import os

    nombre_archivo = os.path.basename(filename)

    match = re.search(
        r'(\d+)\s*x\s*(\d+)',
        nombre_archivo,
        re.IGNORECASE
    )

    if match:

        fs_x = float(match.group(1))
        fs_y = float(match.group(2))

        return (fs_x + fs_y) / 2

    # ----------------------------------
    # Pedir al usuario mediante popup
    # ----------------------------------

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    field_size_cm = simpledialog.askfloat(
        title="Tamaño de campo",
        prompt=(
            "No se pudo determinar el tamaño "
            "de campo a partir del nombre.\n\n"
            "Introduzca el tamaño de campo (cm):"
        ),
        parent=root
    )

    root.destroy()

    if field_size_cm is None:
        raise SystemExit(
            "Operación cancelada por el usuario"
        )

    return field_size_cm

# --- PARÁMETROS DE AJUSTE FOGLIATA (Tabla II) ---
# Basado en Fogliata 2015 para haces 6 MV FFF [5]
FOGLIATA_COEFFS = {
    '6MV_FFF': {'a': 91.0, 'b': 1.53, 'c': 1.15, 'd': -0.0072, 'e': 0.011}
}

def get_axis_profile(net_doses, axis_prefix):
    """Extrae el perfil gestionando gaps y el espaciado correcto."""
    x_coords, y_values = [], []
    spacing = 7.071 if axis_prefix in ['PD', 'ND'] else 5.0
    
    for i in range(1, 67):
        det_id = f"{axis_prefix}{i}"
        if det_id in net_doses:
            pos_mm = (i - 33) * spacing
            x_coords.append(pos_mm)
            y_values.append(net_doses[det_id])
    return np.array(x_coords), np.array(y_values)




def load_ic_profiler_prm(filepath):

    with open(filepath, 'r', encoding='latin-1') as f:
        lines = [line.strip() for line in f.readlines()]

    # -----------------------------------------
    # Localizar secciones
    # -----------------------------------------

    type_idx = next(
        i for i, l in enumerate(lines)
        if l.startswith("TYPE")
    )

    bias_idx = next(
        i for i, l in enumerate(lines)
        if l.startswith("BIAS1")
    )

    cal_indices = [
        i for i, l in enumerate(lines)
        if l.startswith("Calibration")
    ]

    if len(cal_indices) < 2:
        raise ValueError(
            "No se encontró una segunda fila Calibration"
        )

    cal_idx = cal_indices[1]

    data_indices = [
        i for i, l in enumerate(lines)
        if l.startswith("Data:")
    ]

    if len(data_indices) < 2:
        raise ValueError(
            "No hay suficientes frames."
        )

    # -----------------------------------------
    # Cabeceras
    # -----------------------------------------

    cols = lines[type_idx].split('\t')[1:]

    def get_row_series(row_idx, columns):

        vals = lines[row_idx].split('\t')[1:]

        min_len = min(
            len(vals),
            len(columns)
        )

        return pd.Series(
            vals[:min_len],
            index=columns[:min_len]
        ).apply(
            pd.to_numeric,
            errors="coerce"
        )

    bias = get_row_series(
        bias_idx,
        cols
    )

    calib = get_row_series(
        cal_idx,
        cols
    )

    # -----------------------------------------
    # Primer y último frame
    # -----------------------------------------

    last_raw = (
        lines[data_indices[-1]]
        .split('\t')[1:]
    )

    last = np.array(
        [float(x) for x in last_raw]
    )

    time_col_idx = cols.index("TIMETIC")

    last_time = last[time_col_idx]

    corrected_counts = {}

    for det in cols:

        if not any(
            p in det
            for p in ["X", "Y", "PD", "ND"]
        ):
            continue

        idx = cols.index(det)

        corrected_counts[det] = (
            last[idx]
            - last_time * bias.get(det, 0)
        ) * calib.get(det, 1) / 2



    return pd.Series(corrected_counts)


def analyze_fogliata_linear(
        net_doses,
        axis_prefix,
        field_size_cm,
        depth_cm=3.9):

    # =============================
    # Perfil bruto
    # =============================

    x_raw, y_raw = get_axis_profile(
        net_doses,
        axis_prefix
    )

    if len(x_raw) == 0:
        raise ValueError(
            f"No se encontraron datos para {axis_prefix}"
        )

    # =============================
    # Suavizado
    # =============================

##    y_smooth = savgol_filter(
##        y_raw,
##        window_length=3,
##        polyorder=2
##    )

    y_smooth = y_raw

    # =============================
    # Interpolación lineal
    # =============================

    f_lin = interp1d(
        x_raw,
        y_smooth,
        kind="linear",
        fill_value="extrapolate"
    )

    # =============================
    # Renorm Factor Fogliata
    # =============================

    p = FOGLIATA_COEFFS["6MV_FFF"]

    renorm_val = (
        p["a"]
        + p["b"] * field_size_cm
        + p["c"] * depth_cm
    ) / (
        1
        + p["d"] * field_size_cm
        + p["e"] * depth_cm
    )

    cax_raw = float(f_lin(0))

    y_norm = (
        y_smooth / cax_raw
    ) * renorm_val

    f_final = interp1d(
        x_raw,
        y_norm,
        kind="linear",
        fill_value="extrapolate"
    )

    # =============================
    # Malla fina
    # =============================

    x_fine = np.linspace(
        min(x_raw),
        max(x_raw),
        5000
    )

    y_fine = f_final(x_fine)

    center = np.argmin(
        np.abs(x_fine)
    )

    left_x = x_fine[:center]
    left_y = y_fine[:center]

    right_x = x_fine[center:]
    right_y = y_fine[center:]

    # =============================
    # Gaussian Fit
    # =============================

    try:

        p0 = [

            np.max(y_norm) - np.min(y_norm),  # amplitud

            0.0,                              # centro inicial

            40.0,                             # sigma inicial

            np.min(y_norm)                    # offset
        ]

        popt, _ = curve_fit(
            gaussian,
            x_raw,
            y_norm,
            p0=p0,
            maxfev=10000
        )

        A_fit, mu_fit, sigma_fit, C_fit = popt

        gauss_offset = float(mu_fit)

    except Exception:

        gauss_offset = np.nan

    # =============================
    # x = f(dose)
    # =============================

    f_left = interp1d(
        left_y,
        left_x,
        kind="linear",
        bounds_error=False
    )

    f_right = interp1d(
        right_y[::-1],
        right_x[::-1],
        kind="linear",
        bounds_error=False
    )

    # =============================
    # Penumbra + Field Size
    # =============================

    l20 = float(f_left(20))
    l50 = float(f_left(50))
    l80 = float(f_left(80))

    r80 = float(f_right(80))
    r50 = float(f_right(50))
    r20 = float(f_right(20))

    field_size = r50 - l50

    pen_left = abs(
        l80 - l20
    )

    pen_right = abs(
        r20 - r80
    )

    # =============================
    # Field Region Fogliata
    # =============================

    field_center = (
        l50 + r50
    ) / 2

    if field_size_cm < 10:

        fr_factor = 0.60

    else:

        fr_factor = 0.80


    field_region = fr_factor * field_size

    fr_left = (
        field_center
        - field_region/2
    )

    fr_right = (
        field_center
        + field_region/2
    )
    # =============================
    # Unflatness
    # =============================

    cax = float(
        f_final(0)
    )

    dose_fr_avg = (

        f_final(fr_left)
        +
        f_final(fr_right)

    ) / 2

    unflatness = (
        cax / dose_fr_avg
    )

    
    # =============================
    # Slope Fogliata
    # =============================

    half_fs = field_size / 2

    x1_l = -half_fs / 3 + gauss_offset
    x2_l = -2 * half_fs / 3 + gauss_offset

    x1_r = half_fs / 3 + gauss_offset
    x2_r = 2 * half_fs / 3 + gauss_offset


    mask_l = (
        (x_raw >= x2_l)
        &
        (x_raw <= x1_l)
    )

    mask_r = (
        (x_raw >= x1_r)
        &
        (x_raw <= x2_r)
    )


    n_l = np.sum(mask_l)
    n_r = np.sum(mask_r)

    # --------------------------------------------------
    # Caso 1: suficientes detectores -> ajuste lineal
    # --------------------------------------------------

    if min(n_l, n_r) >= 4:

        slope_l, intercept_l = np.polyfit(
            x_raw[mask_l],
            y_norm[mask_l],
            1
        )

        slope_r, intercept_r = np.polyfit(
            x_raw[mask_r],
            y_norm[mask_r],
            1
        )

        D1_l = slope_l * x1_l + intercept_l
        D2_l = slope_l * x2_l + intercept_l

        D1_r = slope_r * x1_r + intercept_r
        D2_r = slope_r * x2_r + intercept_r

        peak_pos = (
            intercept_l
            - intercept_r
        ) / (
            slope_r
            - slope_l
        )

    # --------------------------------------------------
    # Caso 2: pocos detectores -> método clásico
    # --------------------------------------------------

    else:

        D1_l = float(
            f_final(x1_l)
        )

        D2_l = float(
            f_final(x2_l)
        )

        D1_r = float(
            f_final(x1_r)
        )

        D2_r = float(
            f_final(x2_r)
        )

        slope_l = (
            D1_l - D2_l
        ) / (
            x2_l - x1_l
        )

        slope_r = (
            D1_r - D2_r
        ) / (
            x2_r - x1_r
        )

        intercept_l = (
            D1_l
            - slope_l * x1_l
        )

        intercept_r = (
            D1_r
            - slope_r * x1_r
        )

        peak_pos = (
            intercept_l
            - intercept_r
        ) / (
            slope_r
            - slope_l
        )

    # --------------------------------------------------
    # Slope medio
    # --------------------------------------------------

    slope_avg = (
        abs(slope_l)
        +
        abs(slope_r)
    ) / 2
    # =============================
    # Simetría Fogliata
    # =============================

    symmetry = 100.0
    sym_pos = 0.0

    x_sym = np.linspace(
        fr_left,
        0,
        500
    )

    for xx in x_sym:

        dl = float(
            f_final(xx + gauss_offset)
        )

        dr = float(
            f_final(-xx + gauss_offset)
        )

        if dl > 0 and dr > 0:

            s = 100 * max(
                dl / dr,
                dr / dl
            )

            if s > symmetry:

                symmetry = s
                sym_pos = xx

                
    # =============================
    # Resultados
    # =============================

    return {

        "x": x_raw,
        "y": y_norm,
        "f": f_final,

        "field_size": round(field_size,2),
        "field_region": field_region,

        "fr_left": fr_left,
        "fr_right": fr_right,

        "l20": l20,
        "l50": l50,
        "l80": l80,

        "r20": r20,
        "r50": r50,
        "r80": r80,

        "pen_left": round(pen_left,2),
        "pen_right": round(pen_right,2),

        "x1_l": x1_l,
        "x2_l": x2_l,

        "x1_r": x1_r,
        "x2_r": x2_r,

        "D1_l": D1_l,
        "D2_l": D2_l,

        "D1_r": D1_r,
        "D2_r": D2_r,

        "A_fit": A_fit,
        "Mu_fit": mu_fit,
        "Sigma_fit": sigma_fit,
        "C_fit": C_fit,

        "results": {

            "RenormFactor":
                round(
                    renorm_val,
                    2
                ),

            "Unflatness":
                round(
                    unflatness,
                    3
                ),

            "SlopeLeft":
                round(
                    slope_l,
                    4
                ),

            "SlopeRight":
                round(
                    slope_r,
                    4
                ),

            "SlopeAvg":
                round(
                    slope_avg,
                    4
                ),

            "Symmetry":
                round(
                    symmetry,
                    2
                ),

            "PeakPosition":
                round(
                    float(peak_pos),
                    2
                ),
            "GaussianOffset":
                round(
                    gauss_offset,
                    2
                )
        }
    }

# ==========================================
# EJECUCION PRINCIPAL
# ==========================================

if __name__ == "__main__":

    while True:

        root = Tk()
        root.withdraw()

        file_path = askopenfilename(
            title="Selecciona archivo IC Profiler",
            filetypes=[
                ("IC Profiler", "*.prm *.csv"),
                ("Todos", "*.*")
            ]
        )

        root.destroy()

        if not file_path:
            break

        # Ejecuta todo el análisis
        resultados = process_file(file_path)

        # Obtiene 7x7, 10x10 o 20x20
        nombre_archivo = os.path.basename(
            file_path
        )

        campo_actual = obtener_campo(
            nombre_archivo
        )

        # Guarda en CSV
        guardar_resultados_csv(
            {
                "Campo": campo_actual,

                "X_FieldSize": resultados["X"]["field_size"],
                "X_PenumbraLeft": resultados["X"]["pen_left"],
                "X_PenumbraRight": resultados["X"]["pen_right"],
                
                "X_Unflatness":
                    resultados["X"]["Unflatness"],

                "X_SlopeAvg":
                    resultados["X"]["SlopeAvg"],

                "X_Symmetry":
                    resultados["X"]["Symmetry"],

                "X_PeakPosition":
                    resultados["X"]["PeakPosition"],

                "X_GaussianOffset":
                    resultados["X"]["GaussianOffset"],

                "Y_FieldSize": resultados["Y"]["field_size"],
                "Y_PenumbraBottom": resultados["Y"]["pen_left"],
                "Y_PenumbraTop": resultados["Y"]["pen_right"],

                
                "Y_Unflatness":
                    resultados["Y"]["Unflatness"],

                "Y_SlopeAvg":
                    resultados["Y"]["SlopeAvg"],

                "Y_Symmetry":
                    resultados["Y"]["Symmetry"],

                "Y_PeakPosition":
                    resultados["Y"]["PeakPosition"],

                "Y_GaussianOffset":
                    resultados["Y"]["GaussianOffset"]
            }
        )

    print("\nPrograma finalizado.")
