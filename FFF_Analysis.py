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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import os

import re


def atan_edge(x, A, B, x0, C):

    return (
        C
        + A * np.arctan(
            B * (x - x0)
        )
    )

def atan_inverse(
    dose,
    A,
    B,
    x0,
    C
):

    return (
        x0
        +
        np.tan(
            (dose - C) / A
        ) / B
    )

factor = 1000 / 1009


def analyze_profile_geometry(
    net_doses,
    axis_prefix
):
    """
    Analiza la geometría de un perfil de IC Profiler.

    Calcula:
        - Tamaño de campo (50%)
        - Centro del campo
        - Penumbra izquierda y derecha (20-80%)
        - Región útil del campo
        - Ajuste mediante función atan en ambos bordes

    Parámetros
    ----------
    net_doses : pandas.Series
        Perfil corregido acumulado.

    axis_prefix : str
        "X", "Y", "PD" o "ND".

    Returns
    -------
    dict
        Diccionario con toda la información geométrica del perfil.
    """

    # ==========================================================
    # Extracción del perfil bruto
    # ==========================================================

    x_raw, y_raw = get_axis_profile(
        net_doses,
        axis_prefix
    )

    x_raw = x_raw * factor

    if len(x_raw) == 0:
        raise ValueError(
            f"No se encontraron datos para {axis_prefix}"
        )

    # ==========================================================
    # Normalización respecto al eje central (CAX)
    # ==========================================================

    f_lin = interp1d(
        x_raw,
        y_raw,
        kind="linear",
        fill_value="extrapolate"
    )

    cax = float(f_lin(0))

    y_norm = (
        y_raw / cax
    ) * 100

    f = interp1d(
        x_raw,
        y_norm,
        kind="linear",
        fill_value="extrapolate"
    )

    # ==========================================================
    # Generación de una malla fina interpolada
    # ==========================================================

    x_fine = np.linspace(
        np.min(x_raw),
        np.max(x_raw),
        5000
    )

    y_fine = f(x_fine)

    center_idx = np.argmin(
        np.abs(x_fine)
    )

    # ==========================================================
    # Separación del perfil en lado izquierdo y derecho
    # ==========================================================

    left_x = x_fine[:center_idx]
    left_y = y_fine[:center_idx]

    right_x = x_fine[center_idx:]
    right_y = y_fine[center_idx:]

    # ==========================================================
    # Funciones inversas de dosis → posición
    # ==========================================================

    f_left = interp1d(
        left_y,
        left_x,
        bounds_error=False
    )

    f_right = interp1d(
        right_y[::-1],
        right_x[::-1],
        bounds_error=False
    )

    # ==========================================================
    # Posiciones 20%, 50% y 80%
    # ==========================================================

    l20_lin = float(f_left(20))
    l50_lin = float(f_left(50))
    l80_lin = float(f_left(80))

    r80_lin = float(f_right(80))
    r50_lin = float(f_right(50))
    r20_lin = float(f_right(20))

    # ==========================================================
    # Geometría preliminar
    # ==========================================================

    field_size_lin = (
        r50_lin - l50_lin
    )

    field_center_lin = (
        r50_lin + l50_lin
    ) / 2

    # ==========================================================
    # Región de campo preliminar
    # ==========================================================

    if axis_prefix in ["PD", "ND"]:

        fs = field_size_lin / np.sqrt(2)

        if fs <= 100:
            field_region_lin = field_size_lin - 40

        elif fs <= 300:
            field_region_lin = (
                field_size_lin
                - field_size_lin * 0.40
            )

        else:
            field_region_lin = field_size_lin - 120

    else:

        if field_size_lin <= 100:
            field_region_lin = field_size_lin - 20

        elif field_size_lin <= 300:
            field_region_lin = (
                field_size_lin
                - field_size_lin * 0.20
            )

        else:
            field_region_lin = field_size_lin - 60

    fr_left = (
        -field_region_lin / 2
        + field_center_lin
    )

    fr_right = (
        field_region_lin / 2
        + field_center_lin
    )

    # ==========================================================
    # Selección de detectores para ajuste atan
    # ==========================================================

    spacing = (
        7.071
        if axis_prefix in ["PD", "ND"]
        else 5.0
    )

    n_det = 4

    lmin = l50_lin - n_det * spacing
    lmax = l50_lin + n_det * spacing

    rmax = r50_lin - n_det * spacing
    rmin = r50_lin + n_det * spacing

    mask_l = (
        (x_raw >= lmin)
        &
        (x_raw <= lmax)
    )

    mask_r = (
        (x_raw >= rmax)
        &
        (x_raw <= rmin)
    )

    x_l = x_raw[mask_l]
    y_l = y_norm[mask_l]

    x_r = x_raw[mask_r]
    y_r = y_norm[mask_r]

    # ==========================================================
    # Ajuste atan borde izquierdo
    # ==========================================================

    try:

        p0_l = [
            -30,
            0.2,
            l50_lin,
            50
        ]

        popt_l, _ = curve_fit(
            atan_edge,
            x_l,
            y_l,
            p0=p0_l,
            maxfev=10000
        )

        A_l, B_l, X0_l, C_l = popt_l

    except Exception:

        A_l = B_l = X0_l = C_l = np.nan

    # ==========================================================
    # Ajuste atan borde derecho
    # ==========================================================

    try:

        p0_r = [
            30,
            -0.2,
            r50_lin,
            50
        ]

        popt_r, _ = curve_fit(
            atan_edge,
            x_r,
            y_r,
            p0=p0_r,
            maxfev=10000
        )

        A_r, B_r, X0_r, C_r = popt_r

    except Exception:

        A_r = B_r = X0_r = C_r = np.nan

    # ==========================================================
    # Cálculo de posiciones 20%, 50% y 80%
    # mediante los ajustes atan
    # ==========================================================

    l20 = atan_inverse(
        20,
        A_l,
        B_l,
        X0_l,
        C_l
    )

    l50 = atan_inverse(
        50,
        A_l,
        B_l,
        X0_l,
        C_l
    )

    l80 = atan_inverse(
        80,
        A_l,
        B_l,
        X0_l,
        C_l
    )

    r20 = atan_inverse(
        20,
        A_r,
        B_r,
        X0_r,
        C_r
    )

    r50 = atan_inverse(
        50,
        A_r,
        B_r,
        X0_r,
        C_r
    )

    r80 = atan_inverse(
        80,
        A_r,
        B_r,
        X0_r,
        C_r
    )

    # ==========================================================
    # Geometría final
    # ==========================================================

    field_size = (
        r50 - l50
    )

    field_center = (
        l50 + r50
    ) / 2

    pen_left = abs(
        l80 - l20
    )

    pen_right = abs(
        r20 - r80
    )

    # ==========================================================
    # Región útil de campo
    # ==========================================================

    if axis_prefix in ["PD", "ND"]:

        fs = field_size / np.sqrt(2)

        if fs <= 100:

            field_region = (
                field_size - 40
            )

        elif fs <= 300:

            field_region = (
                field_size
                - field_size * 0.40
            )

        else:

            field_region = (
                field_size - 120
            )

    else:

        if field_size <= 100:

            field_region = (
                field_size - 20
            )

        elif field_size <= 300:

            field_region = (
                field_size
                - field_size * 0.20
            )

        else:

            field_region = (
                field_size - 60
            )

    fr_left = (
        -field_region / 2
        + field_center
    )

    fr_right = (
        field_region / 2
        + field_center
    )

    # ==========================================================
    # Resultado
    # ==========================================================

    return {

        "x": x_raw,
        "y": y_norm,
        "f": f,

        "field_size": field_size,
        "field_region": field_region,
        "field_center": field_center,

        "spacing": spacing,

        "fr_left": fr_left,
        "fr_right": fr_right,

        "l20": l20,
        "l50": l50,
        "l80": l80,

        "r20": r20,
        "r50": r50,
        "r80": r80,

        "pen_left": pen_left,
        "pen_right": pen_right,

        "A_l": A_l,
        "B_l": B_l,
        "X0_l": X0_l,
        "C_l": C_l,

        "A_r": A_r,
        "B_r": B_r,
        "X0_r": X0_r,
        "C_r": C_r,

        "x_l": x_l,
        "y_l": y_l,

        "x_r": x_r,
        "y_r": y_r
    }

def obtener_campo(nombre_archivo):
    """
    Obtiene el tamaño de campo del nombre de archivo.

    Ejemplos:
        7x7 6FFF.prm  -> 7x7
        10x10 6MV.prm -> 10x10
        20 x 20.csv   -> 20x20
    """

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


def eliminar_prm():
    """
    Elimina el archivo PRM seleccionado de la lista,
    borra sus resultados almacenados y cierra la
    gráfica actualmente mostrada.
    """

    global canvas_actual

    seleccion = tree.selection()

    if not seleccion:
        return

    file_path = seleccion[0]

    # Eliminar resultados almacenados
    if file_path in resultados_globales:

        del resultados_globales[file_path]

    # Eliminar fila de la tabla
    tree.delete(file_path)

    # Eliminar gráfica mostrada
    if canvas_actual is not None:

        canvas_actual.get_tk_widget().destroy()
        canvas_actual = None

        
def guardar_resultados_csv(
    csv_path,
    resultados,
    campo_actual,
    es_fff
):
    """
    Exporta los resultados de análisis de un archivo PRM a un
    fichero CSV independiente.

    Notas
    -----
    Unidades exportadas:

        - Field Size: cm
        - Penumbras: mm
        - Peak Position: mm
        - Gaussian Offset: mm

    Haces FFF:
        Exporta métricas Fogliata:
            - Field Size
            - Penumbras
            - Unflatness
            - Slope Avg
            - Symmetry
            - Peak Position
            - Gaussian Offset

    Haces filtrados:
        Exporta métricas tipo Profiler:
            - Field Size
            - Penumbras
            - Flatness
            - Symmetry
            - Gaussian Offset

    El CSV generado contiene una única fila con los
    resultados correspondientes al archivo PRM analizado.
    """

    if es_fff:

        # -------------------------
        # Exportación FFF
        # -------------------------

        fila = {

            "Campo": campo_actual,

            "X_FieldSize (cm)":
                resultados["X"]["FieldSize"],

            "X_PenumbraLeft":
                resultados["X"]["PenLeft"],

            "X_PenumbraRight":
                resultados["X"]["PenRight"],

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

            "Y_FieldSize (cm)":
                resultados["Y"]["FieldSize"],

            "Y_PenumbraBottom":
                resultados["Y"]["PenLeft"],

            "Y_PenumbraTop":
                resultados["Y"]["PenRight"],

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

    else:

        # -------------------------
        # Exportación haces filtrados
        # -------------------------

        fila = {

            "Campo": campo_actual,

            "X_FieldSize (cm)":
                resultados["X"]["FieldSize"],

            "X_PenumbraLeft":
                resultados["X"]["PenLeft"],

            "X_PenumbraRight":
                resultados["X"]["PenRight"],

            "X_Flatness":
                resultados["X"]["Flatness"],

            "X_Symmetry":
                resultados["X"]["Symmetry"],

            "Y_FieldSize (cm)":
                resultados["Y"]["FieldSize"],

            "Y_PenumbraBottom":
                resultados["Y"]["PenLeft"],

            "Y_PenumbraTop":
                resultados["Y"]["PenRight"],

            "Y_Flatness":
                resultados["Y"]["Flatness"],

            "Y_Symmetry":
                resultados["Y"]["Symmetry"]
        }

    # Crear DataFrame de una sola fila
    df = pd.DataFrame([fila])

    # Guardar CSV
    df.to_csv(
        csv_path,
        index=False
    )

    print(
        f"[OK] CSV guardado: {csv_path}"
    )
    
def analyze_flat_beam(profile, axis_prefix):
    """
    Análisis IEC para haces filtrados.

    Calcula:
        - Symmetry Point Ratio
        - Flatness-bm
        - Tamaño de campo
        - Penumbras

    Parameters
    ----------
    profile : dict
        Resultado de analyze_profile_geometry().

    Returns
    -------
    dict
        Métricas del perfil.
    """

    # ==========================================================
    # Datos de entrada
    # ==========================================================

    x = profile["x"]
    y = profile["y"]

    field_size = profile["field_size"]
    field_center = profile["field_center"]

    # ==========================================================
    # Región IEC para evaluación
    # ==========================================================

    fs_cm = field_size / 10

    if axis_prefix in ["PD", "ND"]:

        fs = fs_cm / np.sqrt(2)

        if fs <= 10:

            region_cm = (
                fs_cm - 4
            )

        elif fs <= 30:

            region_cm = (
                fs_cm
                - fs_cm * 0.40
            )

        else:

            region_cm = (
                fs_cm - 12
            )

    else:

        if fs_cm <= 10:

            region_cm = (
                fs_cm - 2
            )

        elif fs_cm <= 30:

            region_cm = (
                fs_cm
                - fs_cm * 0.20
            )

        else:

            region_cm = (
                fs_cm - 6
            )

    half_region_mm = (
        region_cm * 10 / 2
    )

    mask = (
        np.abs(
            x - field_center
        )
        <= half_region_mm
    )

    x_region = x[mask]
    y_region = y[mask]

    # ==========================================================
    # Interpolador del perfil
    # ==========================================================

    f_interp = interp1d(
        x,
        y,
        kind="linear",
        fill_value="extrapolate"
    )

    # ==========================================================
    # Symmetry Point Ratio (IEC)
    # ==========================================================

    max_ratio = 1.0

    flat_values = []

    for x_left, d_left in zip(
        x_region,
        y_region
    ):

        x_right = (
            2 * field_center
            - x_left
        )

        d_right = float(
            f_interp(x_right)
        )

        ratio = max(
            d_left / d_right,
            d_right / d_left
        )

        max_ratio = max(
            max_ratio,
            ratio
        )

        flat_values.extend(
            [d_left, d_right]
        )

    symmetry = (
        max_ratio * 100
    )

    # ==========================================================
    # Flatness-bm
    # ==========================================================

    flat_values = np.asarray(
        flat_values
    )

    max_dose = np.max(
        flat_values
    )

    min_dose = np.min(
        flat_values
    )

    flatness = (
        max_dose / min_dose
    ) * 100

    # ==========================================================
    # Resultados
    # ==========================================================

    return {

        "Flatness":
            round(flatness, 3),

        "Symmetry":
            round(symmetry, 3),

        "FieldSize":
            round(
                field_size / 10,
                3
            ),

        "PenLeft":
            round(
                profile["pen_left"],
                3
            ),

        "PenRight":
            round(
                profile["pen_right"],
                3
            )
    }

def calculate_symmetry_stability(
    all_frames,
    eje,
    tolerance=1.0,
    window=10
):

    symmetry = []
    time_s = []

    for _, frame in all_frames.iterrows():

        profile = analyze_profile_geometry(
            frame,
            eje
        )

        res = analyze_flat_beam(
            profile
        )

        symmetry.append(
            res["Symmetry"]
        )

        time_s.append(
            frame["TIMETIC"] / 1e6
        )

    symmetry = np.array(symmetry)
    time_s = np.array(time_s)

    sym_ref = np.median(
        symmetry[-20:]
    )

    deviation = np.abs(
        symmetry - sym_ref
    )

    stability_idx = None

    for i in range(
        len(symmetry) - window + 1
    ):

        if np.all(
            deviation[i:i+window]
            <= tolerance
        ):
            stability_idx = i
            break

    return {
        "symmetry": symmetry,
        "time_s": time_s,
        "mean_symmetry": sym_ref,
        "stability_idx": stability_idx,
        "stability_time_s": (
            time_s[stability_idx]
            if stability_idx is not None
            else np.nan
        )
    }

def process_file(
    file_path,
    show_plot=False,
    return_fig=False
):
    
    resultados_export = {}

    nombre_archivo = os.path.basename(
        file_path
    )

    es_fff = "fff" in nombre_archivo.lower()

    all_frames = load_ic_profiler_prm_all_frames(file_path)


    #######


##    stability_results = {}
##
##    for eje in ["X", "Y", "PD", "ND"]:
##
##        stability_results[eje] = (
##            calculate_symmetry_stability(
##                all_frames,
##                eje,
##                tolerance=1.0,
##                window=10
##            )
##        )
##
##    for eje, res in stability_results.items():
##
##        print(
##            f"{eje}: "
##            f"Simetría media = "
##            f"{res['mean_symmetry']:.2f}%   "
##            f"Tiempo de estabilidad = "
##            f"{res['stability_time_s']:.2f} s"
##        )

##    fig, ax = plt.subplots(
##        figsize=(10,6)
##    )
##
##    for eje, res in stability_results.items():
##
##        ax.plot(
##            res["time_s"],
##            res["symmetry"],
##            label=eje
##        )
##
##        ax.axvline(
##            res["stability_time_s"],
##            linestyle="--",
##            alpha=0.5
##        )
##
##    ax.set_xlabel("Tiempo (s)")
##    ax.set_ylabel("Simetría (%)")
##    ax.legend()
##    ax.grid(True)
##
##    plt.show()
##

    ########




    net_doses = all_frames.iloc[-1]

    field_size_cm = (
        get_field_size_from_filename(
            nombre_archivo
        )
    )



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

    for eje in ["X", "Y", "PD", "ND"]:

        if es_fff:

            data = analyze_fogliata_linear(
                net_doses,
                eje,
                field_size_cm,
                depth_cm=4.08
            )

            res = data["results"]

        else:

            data = analyze_profile_geometry(
                net_doses,
                eje
            )

            res = analyze_flat_beam(
                data, eje
            )

        if eje == "X":

            resultados_export["X"] = {
                **res
            }
            
        if eje == "Y":

            resultados_export["Y"] = {
                **res
            }

        if return_fig:

            # ==========================================
            # GRAFICA 2x2
            # ==========================================
                    
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
            field_center = data["field_center"]

            pen_left = data["pen_left"]
            pen_right = data["pen_right"]
            


            # =========================
            # Perfil completo
            # =========================

            x_plot = np.linspace(
                np.min(x),
                np.max(x),
                5000
            )

            y_plot = f(x_plot)



            A_l = data["A_l"]
            B_l = data["B_l"]
            X0_l = data["X0_l"]
            C_l = data["C_l"]

            A_r = data["A_r"]
            B_r = data["B_r"]
            X0_r = data["X0_r"]
            C_r = data["C_r"]

            x_l = data["x_l"]
            y_l = data["y_l"]

            x_r = data["x_r"]
            y_r = data["y_r"]

            y_atan_l = atan_edge(
                x_plot,
                A_l,
                B_l,
                X0_l,
                C_l
            )

            y_atan_r = atan_edge(
                x_plot,
                A_r,
                B_r,
                X0_r,
                C_r
            )
            
            # =====================
            # Left atan fit
            # =====================

            mask_plot_l = (
                (x_plot >= np.min(x_l))
                &
                (x_plot <= np.max(x_l))
            )

            ax.plot(
                x_plot[mask_plot_l],
                y_atan_l[mask_plot_l],
                color="darkgreen",
                linestyle="--",
                linewidth=0.8,
                alpha=0.8,
                zorder = 50
            )

            


            # =====================
            # Right atan fit
            # =====================

            mask_plot_r = (
                (x_plot >= np.min(x_r))
                &
                (x_plot <= np.max(x_r))
            )

            ax.plot(
                x_plot[mask_plot_r],
                y_atan_r[mask_plot_r],
                color="darkgreen",
                linestyle="--",
                linewidth=0.8,
                alpha=0.8,
                zorder = 50
            )


            # Curva completa naranja

            ax.plot(
                x_plot,
                y_plot,
                color="darkorange",
                lw=1.5
            )

    ##        ax.scatter(
    ##            x,
    ##            y,
    ##            color="black",
    ##            s=8,
    ##            alpha=0.6,
    ##            zorder=5
    ##        )

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
                ha="center",
                fontsize=8
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
                ha="center",
                fontsize=8
            )

            if es_fff:


                # =========================
                # Slope points
                # =========================
            

                x1_l = data["x1_l"]
                x2_l = data["x2_l"]

                x1_r = data["x1_r"]
                x2_r = data["x2_r"]

                D1_l = data["D1_l"]
                D2_l = data["D2_l"]

                D1_r = data["D1_r"]
                D2_r = data["D2_r"]
                    
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
                    f"Left Slope = {res['SlopeLeft']:.4f} mm⁻¹\n"
                    f"Slope Avg = {res['SlopeAvg']:.4f} mm⁻¹\n"
                    f"Unflatness = {res['Unflatness']:.3f}"
                )

                ax.text(
                    -0.05,
                    1.1,
                    txt_left,
                    transform=ax.transAxes,
                    va="top",
                    zorder=50,
                    fontsize=8,
                    bbox=dict(
                        facecolor="wheat",
                        alpha=0.90
                    )
                )

                # =========================
                # Cuadro derecho
                # =========================

                txt_right = (
                    f"Beam Center = {field_center:.2f} mm\n"
                    f"Right Penumbra = {pen_right:.2f} mm\n"
                    f"Right Slope = {res['SlopeRight']:.4f} mm⁻¹\n"
                    f"Symmetry = {res['Symmetry']:.2f} %\n"
                    f"Peak Position = {res['PeakPosition']:.2f} mm"
                )
                ax.text(
                    0.68,
                    1.1,
                    txt_right,
                    transform=ax.transAxes,
                    va="top",
                    fontsize=8,
                    zorder=50,
                    bbox=dict(
                        facecolor="wheat",
                        alpha=0.90
                    )
                )


            else:


                # =========================
                # Cuadro izquierdo
                # =========================

                txt_left = (
                    f"Left Penumbra = {pen_left:.2f} mm\n"
                    f"Flatness = {res['Flatness']:.2f} %"
                )

                ax.text(
                    -0.05,
                    1.1,
                    txt_left,
                    transform=ax.transAxes,
                    va="top",
                    zorder=50,
                    fontsize=8,
                    bbox=dict(
                        facecolor="wheat",
                        alpha=0.90
                    )
                )

                # =========================
                # Cuadro derecho
                # =========================

                txt_right = (
                    f"Beam Center = {field_center:.2f} mm\n"
                    f"Right Penumbra = {pen_right:.2f} mm\n"
                    f"Symmetry = {res['Symmetry']:.2f} %"
                )
                ax.text(
                    0.68,
                    1.1,
                    txt_right,
                    transform=ax.transAxes,
                    va="top",
                    fontsize=8,
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
                f"Eje {eje}",
                fontsize=8
            )

            ax.set_xlabel(
                "Off-axis (mm)",
                fontsize=9
            )

            ax.set_ylabel(
                "Dose (%)",
                fontsize=9
            )

            ax.tick_params(
                labelsize=8
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
            f"IC Profiler - Elekta",
            fontsize=10
        )
    
##    plt.tight_layout()
##    fig, axs = plt.subplots(
##        2,
##        2,
##        figsize=(12, 10)
##    )
##    


    if return_fig:
        
        return fig

    print("\nRESULTADOS EXPORT")
    print(resultados_export.keys())

    return resultados_export

##
##    return {
##        "results": resultados_export,
##        "fig": fig
##    }
    
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
    
    for i in range(1, 65):
        det_id = f"{axis_prefix}{i}"
        if det_id in net_doses:
            pos_mm = (i - 33) * spacing
            x_coords.append(pos_mm)
            y_values.append(net_doses[det_id])
    return np.array(x_coords), np.array(y_values)



def load_ic_profiler_prm_all_frames(filepath):

    with open(filepath, 'r', encoding='latin-1') as f:
        lines = [line.strip() for line in f.readlines()]

    type_idx = next(
        i for i, l in enumerate(lines)
        if l.startswith("TYPE")
    )

    bias_idx = next(
        i for i, l in enumerate(lines)
        if l.startswith("BIAS1")
    )

    nominal_gain = float(
        next(
            l for l in lines
            if l.startswith("Nominal Gain")
        ).split('\t')[1]
    )

    cal_indices = [
        i for i, l in enumerate(lines)
        if l.startswith("Calibration")
    ]

    cal_idx = cal_indices[1]

    data_indices = [
        i for i, l in enumerate(lines)
        if l.startswith("Data:\t")
    ]

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

    detectors = [
        c for c in cols
        if any(
            p in c
            for p in ["X", "Y", "PD", "ND"]
        )
    ]

    rows = []

    time_col_idx = cols.index("TIMETIC")

    for idx in data_indices:

        raw = lines[idx].split('\t')[1:]

        values = np.array(
            [float(x) for x in raw]
        )

        timetic = values[time_col_idx]

        corrected = {}

        for det in detectors:

            col_idx = cols.index(det)

            corrected[det] = (
                values[col_idx]
                - timetic * bias.get(det, 0)
            ) * calib.get(det, 1) / nominal_gain

        corrected["TIMETIC"] = timetic
        corrected["UPDATE#"] = values[0]

        rows.append(corrected)

    return pd.DataFrame(rows)



def analyze_fogliata_linear(
        net_doses,
        axis_prefix,
        field_size_cm,
        depth_cm=4.08):

    # =============================
    # Perfil bruto
    # =============================

    x_raw, y_raw = get_axis_profile(
        net_doses,
        axis_prefix
    )

    x_raw = x_raw * factor

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

##    # =============================
##    # Penumbra + Field Size
##    # =============================
##
##    l20 = float(f_left(20))
##    l50 = float(f_left(50))
##    l80 = float(f_left(80))
##
##    r80 = float(f_right(80))
##    r50 = float(f_right(50))
##    r20 = float(f_right(20))



    # ==========================================================
    # Selección de detectores para ajuste atan
    # ==========================================================

    spacing = (
        7.071
        if axis_prefix in ["PD", "ND"]
        else 5.0
    )

    n_det = 3


    l50_lin = float(f_left(50))
    r50_lin = float(f_right(50))

    lmin = l50_lin - n_det * spacing
    lmax = l50_lin + n_det * spacing

    rmax = r50_lin - n_det * spacing
    rmin = r50_lin + n_det * spacing

    mask_l = (
        (x_raw >= lmin)
        &
        (x_raw <= lmax)
    )

    mask_r = (
        (x_raw >= rmax)
        &
        (x_raw <= rmin)
    )

    x_l = x_raw[mask_l]
    y_l = y_norm[mask_l]

    x_r = x_raw[mask_r]
    y_r = y_norm[mask_r]

    # ==========================================================
    # Ajuste atan borde izquierdo
    # ==========================================================

    try:

        p0_l = [
            -30,
            0.2,
            l50_lin,
            50
        ]

        popt_l, _ = curve_fit(
            atan_edge,
            x_l,
            y_l,
            p0=p0_l,
            maxfev=10000
        )

        A_l, B_l, X0_l, C_l = popt_l

    except Exception:

        A_l = B_l = X0_l = C_l = np.nan

    # ==========================================================
    # Ajuste atan borde derecho
    # ==========================================================

    try:

        p0_r = [
            30,
            -0.2,
            r50_lin,
            50
        ]

        popt_r, _ = curve_fit(
            atan_edge,
            x_r,
            y_r,
            p0=p0_r,
            maxfev=10000
        )

        A_r, B_r, X0_r, C_r = popt_r

    except Exception:

        A_r = B_r = X0_r = C_r = np.nan

    # ==========================================================
    # Cálculo de posiciones 20%, 50% y 80%
    # mediante los ajustes atan
    # ==========================================================

    l20 = atan_inverse(
        20,
        A_l,
        B_l,
        X0_l,
        C_l
    )

    l50 = atan_inverse(
        50,
        A_l,
        B_l,
        X0_l,
        C_l
    )

    l80 = atan_inverse(
        80,
        A_l,
        B_l,
        X0_l,
        C_l
    )

    r20 = atan_inverse(
        20,
        A_r,
        B_r,
        X0_r,
        C_r
    )

    r50 = atan_inverse(
        50,
        A_r,
        B_r,
        X0_r,
        C_r
    )

    r80 = atan_inverse(
        80,
        A_r,
        B_r,
        X0_r,
        C_r
    )


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
    # Gaussian Fit
    # =============================

    field_mask = (
        (x_raw >= fr_left)
        &
        (x_raw <= fr_right)
    )

    x_fit = x_raw[field_mask]
    y_fit = y_norm[field_mask]

    try:

        p0 = [

            np.max(y_norm) - np.min(y_norm),  # amplitud

            0.0,                              # centro inicial

            40.0,                             # sigma inicial

            np.min(y_norm)                    # offset
        ]

        popt, _ = curve_fit(
            gaussian,
            x_fit,
            y_fit,
            p0=p0,
            maxfev=10000
        )

        A_fit, mu_fit, sigma_fit, C_fit = popt

        gauss_offset = float(mu_fit)

    except Exception:

        gauss_offset = np.nan
##
##
##    # =============================
##    # Centroide
##    # =============================
##
##    try:
##
##        mask = y_norm >= 50
##
##        x_field = x_raw[mask]
##        y_field = y_norm[mask]
##
##        gauss_offset = np.sum(
##            x_field * y_field
##        ) / np.sum(y_field)
##
##    except Exception:
##
##        gauss_offset = np.nan

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

    x1_l = -half_fs / 3 + field_center
    x2_l = -2 * half_fs / 3 + field_center

    x1_r = half_fs / 3 + field_center
    x2_r = 2 * half_fs / 3 + field_center


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
            f_final(xx + field_center)
        )

        dr = float(
            f_final(-xx + field_center)
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

        "field_region": field_region,
        "field_center": field_center,
        "field_size": field_size,

        "fr_left": fr_left,
        "fr_right": fr_right,

        "l20": l20,
        "l50": l50,
        "l80": l80,

        "r20": r20,
        "r50": r50,
        "r80": r80,

        "pen_left": pen_left,
        "pen_right": pen_right,

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

        "A_l": A_l,
        "B_l": B_l,
        "X0_l": X0_l,
        "C_l": C_l,

        "A_r": A_r,
        "B_r": B_r,
        "X0_r": X0_r,
        "C_r": C_r,

        "x_l": x_l,
        "y_l": y_l,

        "x_r": x_r,
        "y_r": y_r,

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
                ),
            
            "FieldSize":
                round(
                    field_size / 10,
                    3
                ),

            "PenLeft":
                round(
                    pen_left,
                    3
                ),

            "PenRight":
                round(
                    pen_right,
                    3
                )
        }
    }

def cargar_prm():

    files = filedialog.askopenfilenames(
        title="Seleccionar PRM",
        filetypes=[
            ("PRM", "*.prm")
        ]
    )

    for file_path in files:

        nombre = os.path.basename(
            file_path
        )

        es_fff = (
            "fff"
            in nombre.lower()
        )

        tipo = (
            "FFF"
            if es_fff
            else "FLAT"
        )

        # análisis
        resultados[file_path] = (
            process_file(
                file_path,
                show_plot=False
            )
        )

        tree.insert(
            "",
            "end",
            iid=file_path,
            values=(
                nombre,
                tipo
            )
        )

def visualizar(event):

    item = tree.selection()

    if not item:
        return

    file_path = item[0]

    mostrar_figura(
        file_path
    )

def exportar_csv():

    carpeta = filedialog.askdirectory(
        title="Seleccionar carpeta destino"
    )

    if not carpeta:
        return

    for file_path, datos in resultados.items():

        exportar_resultados_csv(
            file_path,
            datos,
            carpeta
        )
        
# ==========================================
# EJECUCION PRINCIPAL GUI
# ==========================================

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import os

resultados_globales = {}

ultima_carpeta_prm = ""

canvas_actual = None

def cargar_prm():

    global ultima_carpeta_prm

    files = filedialog.askopenfilenames(
        title="Seleccionar archivos PRM",
        filetypes=[
            ("IC Profiler PRM", "*.prm")
        ]
    )

    if not files:
        return

# Guarda la carpeta para usarla después

    ultima_carpeta_prm = os.path.dirname(
        files[0]
    )

    for file_path in files:

        if file_path in resultados_globales:
            continue

        try:
            resultados_globales[file_path] = process_file(
                file_path,
                show_plot=False
            )

            nombre = os.path.basename(
                file_path
            )

            tipo = (
                "FFF"
                if "fff" in nombre.lower()
                else "FLAT"
            )

            tree.insert(
                "",
                "end",
                iid=file_path,
                values=(
                    nombre,
                    tipo
                )
            )

        except Exception as e:

            print(
                f"Error analizando "
                f"{file_path}: {e}"
            )


def visualizar():

    seleccion = tree.selection()

    if not seleccion:
        return

    file_path = seleccion[0]

    mostrar_figura(file_path)

def actualizar_grafica(event):

    seleccion = tree.selection()

    if not seleccion:
        return

    file_path = seleccion[0]

    mostrar_figura(file_path)



def mostrar_figura(file_path):

    global canvas_actual

    # eliminar gráfica previa
    if canvas_actual is not None:

        canvas_actual.get_tk_widget().destroy()

    # generar figura SIN mostrarla
    fig = process_file(
        file_path,
        show_plot=False,
        return_fig=True
    )

    canvas_actual = FigureCanvasTkAgg(
        fig,
        master=frame_plot
    )

    canvas_actual.draw()

    canvas_actual.get_tk_widget().pack(
        fill="both",
        expand=True
    )


##    global canvas_actual
##
##    if canvas_actual:
##        canvas_actual.get_tk_widget().destroy()
##
##    fig = resultados_globales[file_path]["fig"]
##
##    canvas_actual = FigureCanvasTkAgg(
##        fig,
##        master=frame_plot
##    )
##
##    canvas_actual.draw()
##
##    canvas_actual.get_tk_widget().pack(
##        fill="both",
##        expand=True
##    )

##    if file_path not in resultados_globales:
##        return
##
##    fig = (
##        resultados_globales[file_path]
##        ["fig"]
##    )
##
##    fig.canvas.manager.show()
##    fig.canvas.draw_idle()

def exportar_csv():

    if not resultados_globales:
        return

    carpeta = filedialog.askdirectory(
        title="Seleccionar carpeta destino",
        initialdir=ultima_carpeta_prm
    )

    if not carpeta:
        return

    for file_path, resultados in (
        resultados_globales.items()
    ):

        nombre_archivo = (
            os.path.basename(
                file_path
            )
        )

        campo_actual = (
            obtener_campo(
                nombre_archivo
            )
        )

        es_fff = (
            "fff"
            in nombre_archivo.lower()
        )

        csv_name = os.path.join(
            carpeta,
            os.path.splitext(
                nombre_archivo
            )[0]
            + ".csv"
        )

        guardar_resultados_csv(
            csv_name,
            resultados,
            campo_actual,
            es_fff
        )

    print(
        "\nTodos los CSV exportados."
    )
# ------------------------------------------
# Ventana principal
# ------------------------------------------

root = tk.Tk()

root.title(
    "IC Profiler Analyzer"
)

root.geometry(
    "1400x800"
)

# ------------------------------------------
# Layout principal
# ------------------------------------------

main_frame = tk.Frame(root)

main_frame.pack(
    fill="both",
    expand=True
)

# ------------------------------------------
# Panel izquierdo
# ------------------------------------------

frame_left = tk.Frame(
    main_frame,
    width=300
)

frame_left.pack(
    side="left",
    fill="y",
    padx=5,
    pady=5
)

# ------------------------------------------
# Panel derecho (gráficas)
# ------------------------------------------

frame_plot = tk.Frame(
    main_frame
)

frame_plot.pack(
    side="right",
    fill="both",
    expand=True,
    padx=5,
    pady=5
)

# ------------------------------------------
# Botón cargar
# ------------------------------------------

btn_load = tk.Button(
    frame_left,
    text="Cargar archivos PRM",
    command=cargar_prm
)

btn_load.pack(
    pady=10,
    fill="x"
)


# ------------------------------------------
# Botón eliminar
# ------------------------------------------

btn_delete = tk.Button(
    frame_left,
    text="Eliminar seleccionado",
    command=eliminar_prm,
    bg="tomato"
)

btn_delete.pack(
    pady=5,
    fill="x"
)
# ------------------------------------------
# Lista archivos
# ------------------------------------------

tree = ttk.Treeview(
    frame_left,
    columns=(
        "archivo",
        "tipo"
    ),
    show="headings",
    height=25
)

tree.heading(
    "archivo",
    text="Archivo"
)

tree.heading(
    "tipo",
    text="Tipo"
)

tree.column(
    "archivo",
    width=180
)

tree.column(
    "tipo",
    width=70,
    anchor="center"
)

tree.pack(
    fill="both",
    expand=True
)

# ------------------------------------------
# Mostrar gráfica al seleccionar
# ------------------------------------------

tree.bind(
    "<<TreeviewSelect>>",
    actualizar_grafica
)

# ------------------------------------------
# Botón exportar
# ------------------------------------------

btn_export = tk.Button(
    frame_left,
    text="Exportar resultados CSV",
    command=exportar_csv,
    bg="lightgreen"
)

btn_export.pack(
    pady=10,
    fill="x"
)

# ------------------------------------------
# Ejecutar GUI
# ------------------------------------------

root.mainloop()
