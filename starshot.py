import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from scipy.signal import find_peaks
from scipy.optimize import curve_fit

import os


# ==========================================
# CARGA IMAGEN
# ==========================================

def load_image(filename):

    img = np.array(Image.open(filename))

    if img.ndim == 3:
        img = img[:,:,0]

    img = img.astype(np.float32)

    img = (img-img.min())/(img.max()-img.min())

    img = (255*img).astype(np.uint8)

    return img


# ==========================================
# SELECCION ARCHIVO
# ==========================================

def select_file():

    root = tk.Tk()
    root.withdraw()

    return filedialog.askopenfilename(
        title="Star Shot",
        filetypes=[
            ("Imagenes","*.tif *.tiff *.png *.jpg")
        ]
    )


# ==========================================
# ROI
# ==========================================

def select_roi(image):

    roi = cv2.selectROI(
        "Seleccion ROI",
        image,
        False
    )

    cv2.destroyAllWindows()

    x,y,w,h = roi

    return image[
        int(y):int(y+h),
        int(x):int(x+w)
    ]


# ==========================================
# CENTRO + RADIO + THRESHOLD
# ==========================================

def setup_analysis(image):

    center = [
        image.shape[1]//2,
        image.shape[0]//2
    ]

    radius = min(
        image.shape
    )//4

    threshold = 120

    def redraw():

        _, binary = cv2.threshold(
            image,
            threshold,
            255,
            cv2.THRESH_BINARY_INV
        )

        display = cv2.cvtColor(
            binary,
            cv2.COLOR_GRAY2BGR
        )

        # ---- TEXTO ----

        # fondo gris
        cv2.rectangle(
            display,
            (5, 5),
            (240, 100),
            (80, 80, 80),   # gris oscuro
            -1              # relleno
        )

        # borde opcional
        cv2.rectangle(
            display,
            (5, 5),
            (240, 100),
            (150, 150, 150),
            1
        )

        cv2.putText(
            display,
            "Click: centro",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            "+/W: radio +",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            "-/S: radio -",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        # ---- CIRCULO ----
        cv2.circle(
            display,
            tuple(center),
            radius,
            (0,255,0),
            2
        )

        cv2.drawMarker(
            display,
            tuple(center),
            (0,0,255),
            cv2.MARKER_CROSS,
            20,
            2
        )

        cv2.imshow(
            "Threshold/Centro",
            display
        )

    def mouse(event,x,y,flags,param):

        if event == cv2.EVENT_LBUTTONDOWN:

            center[0]=x
            center[1]=y

            redraw()

    cv2.namedWindow(
        "Threshold/Centro"
    )

    cv2.setMouseCallback(
        "Threshold/Centro",
        mouse
    )

    cv2.createTrackbar(
        "Threshold",
        "Threshold/Centro",
        threshold,
        255,
        lambda x: None
    )

    redraw()

    while True:

        threshold = cv2.getTrackbarPos(
            "Threshold",
            "Threshold/Centro"
        )

        redraw()

        key = cv2.waitKeyEx(30)

        if key == 13:
            break

        elif key == 27:
            return None

        elif key in (ord("+"), ord("w")):
            radius += 5

        elif key in (ord("-"), ord("s")):
            radius = max(
                20,
                radius-5
            )

    cv2.destroyAllWindows()

    return center,radius,threshold


# ==========================================
# PERFIL ANGULAR MEJORADO
# ==========================================

def angular_profile(
    image,
    center,
    radius
):

    cx, cy = center

    angles = np.linspace(
        0,
        360,
        1440,
        endpoint=False
    )

    values = []

    for angle in angles:

        theta = np.deg2rad(angle)

        samples = []

        for dr in range(-10, 11):

            r = radius + dr

            x = cx + r * np.cos(theta)
            y = cy - r * np.sin(theta)

            ix = int(round(x))
            iy = int(round(y))

            if (
                0 <= ix < image.shape[1]
                and
                0 <= iy < image.shape[0]
            ):

                samples.append(
                    image[iy, ix]
                )

        if len(samples) > 0:

            values.append(
                np.mean(samples)
            )

        else:

            values.append(0)

    return angles, np.array(values)



# ==========================================
# PICOS (CENTROIDE + SUAVIZADO)
# ==========================================

from scipy.ndimage import gaussian_filter1d, label
##from scipy.optimize import curve_fit


##def gaussian(x, a, x0, sigma, b):
##
##    return (
##        a *
##        np.exp(
##            -((x - x0) ** 2) /
##            (2 * sigma * sigma)
##        )
##        + b
##    )


def find_peak_angles(
    angles,
    profile
):

    signal = gaussian_filter1d(
        profile.astype(float),
        sigma=2
    )

    threshold = (
        np.min(signal)
        + np.max(signal)
    ) / 2

    # detectar montañas
    mask = signal > threshold

    labels, nregions = label(mask)

    centers = []

    for region in range(1, nregions + 1):
        idx = labels == region

        theta = angles[idx]
        peak = signal[idx]


        if len(theta) < 5:
            continue

        peak = peak - np.min(peak)

        if np.max(peak) <= 0:
            continue

        # Centroide ponderado

        try:

            center = np.sum(
                theta * peak
            ) / np.sum(peak)

            centers.append(center)

        except:

            centers.append(
                theta[np.argmax(peak)]
            )

    centers = np.array(centers)

    centers = np.sort(centers)

    # fusionar la montaña que cruza 0°/360°

    if (
        len(centers) > 1
        and centers[0] < 20
        and centers[-1] > 340
    ):

        merged = (
            centers[0]
            + (centers[-1] - 360)
        ) / 2

        if merged < 0:
            merged += 360

        centers = centers[1:-1]

        centers = np.insert(
            centers,
            0,
            merged
        )

    return centers
# ==========================================
# RECTAS OPUESTAS
# ==========================================

def build_lines(center, radius, angles, tolerance=10):

    cx, cy = center

    used = np.zeros(len(angles), dtype=bool)

    lines = []

    for i, a in enumerate(angles):

        if used[i]:
            continue
        target = (a + 180) % 360

        delta = np.abs(
            ((angles - target + 180) % 360) - 180
        )

        idx = np.argmin(delta)

        if delta[idx] > tolerance:
            continue

        used[i] = True
        used[idx] = True

        t1 = np.deg2rad(a)
        t2 = np.deg2rad(angles[idx])

        x1 = cx + radius * np.cos(t1)
        y1 = cy - radius * np.sin(t1)

        x2 = cx + radius * np.cos(t2)
        y2 = cy - radius * np.sin(t2)

        lines.append(
            (x1, y1, x2, y2)
        )

    return lines


# ==========================================
# DISPLAY
# ==========================================

def clip_line_to_image(x1, y1, x2, y2, width, height):

    points = []

    dx = x2 - x1
    dy = y2 - y1

    # x = 0
    if abs(dx) > 1e-12:
        t = (0 - x1) / dx
        y = y1 + t * dy

        if 0 <= y <= height:
            points.append((0, y))

    # x = width
    if abs(dx) > 1e-12:
        t = (width - x1) / dx
        y = y1 + t * dy

        if 0 <= y <= height:
            points.append((width, y))

    # y = 0
    if abs(dy) > 1e-12:
        t = (0 - y1) / dy
        x = x1 + t * dx

        if 0 <= x <= width:
            points.append((x, 0))

    # y = height
    if abs(dy) > 1e-12:
        t = (height - y1) / dy
        x = x1 + t * dx

        if 0 <= x <= width:
            points.append((x, height))

    # eliminar duplicados
    unique = []

    for p in points:

        duplicate = False

        for q in unique:

            if np.hypot(
                p[0] - q[0],
                p[1] - q[1]
            ) < 1e-6:

                duplicate = True
                break

        if not duplicate:
            unique.append(p)

    if len(unique) < 2:
        return None

    return unique[0], unique[1]

def display_results(
    image,
    center,
    radius,
    profile,
    theta,
    angles,
    filename
):

    lines = build_lines(
        center,
        radius,
        angles
    )

    star_center, star_radius = (
        calculate_starshot_center(
            lines,
            center
        )
    )

    fig,ax = plt.subplots(
        1,
        2,
        figsize=(15,7)
    )

    ax[0].imshow(
        image,
        cmap="jet"
    )


##    circle = plt.Circle(
##        center,
##        radius,
##        fill=False,
##        color="lime"
##    )
##
##    ax[0].add_patch(circle)

##    star_circle = plt.Circle(
##        (
##            star_center[0],
##            star_center[1]
##        ),
##        star_radius,
##        fill=False,
##        color="cyan",
##        linewidth=3
##    )
##
##    ax[0].add_patch(
##        star_circle
##    )

##    ax[0].scatter(
##        star_center[0],
##        star_center[1],
##        c="yellow",
##        s=100,
##        zorder=10
##    )


    h, w = image.shape

    for x1, y1, x2, y2 in lines:

        pts = clip_line_to_image(
            x1,
            y1,
            x2,
            y2,
            w-1,
            h-1
        )

        if pts is None:
            continue

        p1, p2 = pts

        ax[0].plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            'r',
            lw=2
        )
    
##    ax[0].scatter(
##        center[0],
##        center[1],
##        c='yellow'
##    )
        
    short_name = os.path.basename(filename)

    ax[0].set_title(
        f"{short_name}\n"
        "Star Shot With Spokes"
    )

    
    ax[0].set_xticks([])
    ax[0].set_yticks([])



    # ==========================================
    # PANEL DERECHO (TIPO SUNCHECK)
    # ==========================================

    ax[1].imshow(
        image,
        cmap="gray"
    )

    # mismas rectas que en la izquierda

    for x1l, y1l, x2l, y2l in lines:

        ax[1].plot(
            [x1l, x2l],
            [y1l, y2l],
            color="yellow",
            lw=2
        )

    # círculo minimax

    star_circle = plt.Circle(
        (
            star_center[0],
            star_center[1]
        ),
        star_radius,
        fill=False,
        color="red",
        linewidth=3
    )

    ax[1].add_patch(
        star_circle
    )



    # ---------------------------------
    # zoom centrado en el StarShot
    # ---------------------------------

    zoom_radius = max(
        star_radius * 5,
        10
    )

    ax[1].set_xlim(
        star_center[0] - zoom_radius,
        star_center[0] + zoom_radius
    )

    ax[1].set_ylim(
        star_center[1] + zoom_radius,
        star_center[1] - zoom_radius
    )

    mm_per_pixel = 25.4 / 300

    radius_mm = (
        star_radius *
        mm_per_pixel
    )

    diameter_mm = (
        2 * radius_mm
    )

    ax[1].set_title(
        f"Star Shot Optimal Circle\n"
        f"Diameter = {diameter_mm:.3f} mm"
    )

    ax[1].set_xticks([])
    ax[1].set_yticks([])

    

    print()
    print("================================")
    print("STARSHOT")
    print("================================")

    print(
        f"Center X = {star_center[0]:.2f} px"
    )

    print(
        f"Center Y = {star_center[1]:.2f} px"
    )

    print(
        f"Radius = {radius_mm:.3f} mm"
    )

    print(
        f"Diameter = {diameter_mm:.3f} mm"
    )

    print("================================")

    
    plt.show()

# ==========================================
# RECTAS -> AX + BY + C = 0
# ==========================================

def lines_to_equations(lines):

    equations = []

    for x1, y1, x2, y2 in lines:

        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2

        norm = np.sqrt(A*A + B*B)

        if norm == 0:
            continue

        A /= norm
        B /= norm
        C /= norm

        equations.append((A, B, C))

    return equations


# ==========================================
# STARSHOT MINIMAX
# ==========================================

from scipy.optimize import minimize


def calculate_starshot_center(
    lines,
    center_guess
):

    equations = lines_to_equations(lines)

    def objective(p):

        x, y = p

        distances = []

        for A, B, C in equations:

            d = abs(
                A * x +
                B * y +
                C
            )

            distances.append(d)

        return np.max(distances)

    result = minimize(
        objective,
        center_guess,
        method="Nelder-Mead"
    )

    center = result.x

    radius = objective(center)

    return center, radius




# ==========================================
# MAIN
# ==========================================

def main():

    filename = select_file()

    if not filename:
        return

    image = load_image(
        filename
    )
##
##    roi = select_roi(
##        image
##    )

    roi = image

    config = setup_analysis(
        roi
    )

    if config is None:
        return

    center,radius,threshold = config

    _,binary = cv2.threshold(
        roi,
        threshold,
        255,
        cv2.THRESH_BINARY_INV
    )

    theta,profile = angular_profile(
        binary,
        center,
        radius
    )

    peak_angles = find_peak_angles(
        theta,
        profile
    )

    print()

    display_results(
        roi,
        center,
        radius,
        profile,
        theta,
        peak_angles,
        filename
    )


if __name__=="__main__":
    main()
