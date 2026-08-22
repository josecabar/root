import tkinter as tk
import numpy as np
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import cv2
from matplotlib.widgets import RectangleSelector



# Constants
RUTA_DEFAULT = '.'  # Default path

# Create the main window
ventana_principal = tk.Tk()
ventana_principal.title("Images .his Analyzer")

# Global variables
ruta_archivos = ''
lista_archivos = []
indice_actual = 0
roi1 = None  # ROI1
roi2 = None  # Segunda ROI
canvas_image = None
valores_media_roi1 = []  # For storing the means of ROI1
desplazamientos = []  # For storing the values of shifts of ROI2
valores_media_roi1_filtered = []  
desplazamientos_filtered = []  

ms_value = 0
delta_ms_values = []
escala_pixel = 0.521 # mm/pixel
tamano_filtro = 25

delta_ms_IN_pos = []
delta_ms_IN_neg = []
delta_ms_OUT_pos = []
delta_ms_OUT_neg = []
delta_ms_ON = []
delta_ms_OFF = []

umbral_estado_minimo = 0
umbral_estado_maximo = 0
umbral_estado_minimo2 = 0
tolerancia = 1.0 # valor en mm de tolerancia de ExacTrac
tolerancia1 = 0.1 # valor entre 0 y 1 mm del umbral de tolerancia de kV



labels = []

# Variable to store the current path
ruta_actual = tk.StringVar()
ruta_actual.set(RUTA_DEFAULT)

etiqueta_ruta_actual = tk.Label(ventana_principal, textvariable=ruta_actual)
etiqueta_ruta_actual.pack()

marco_botones = tk.Frame(ventana_principal)
marco_botones.pack(side=tk.LEFT, padx=10)

etiqueta_imagen_actual = tk.Label(ventana_principal)
etiqueta_imagen_actual.pack(side=tk.RIGHT)

texto_imagen = tk.StringVar()
texto_imagen.set("Image ... of ...")
boton_imagen = tk.Label(marco_botones, textvariable=texto_imagen)
boton_imagen.pack()

boton_seleccionar_carpeta = tk.Button(marco_botones, text="Select folder", command=lambda: seleccionar_carpeta())
boton_seleccionar_carpeta.pack()

boton_anterior = tk.Button(marco_botones, text="Previous", command=lambda: mostrar_anterior())
boton_anterior.pack()
boton_siguiente = tk.Button(marco_botones, text="Next", command=lambda: mostrar_siguiente())
boton_siguiente.pack()

entrada_numero_imagen = tk.Entry(marco_botones)
entrada_numero_imagen.pack()
entrada_numero_imagen.bind('<Return>', lambda event: ir_a_imagen())

boton_ir_a_imagen = tk.Button(marco_botones, text="Go to image", command=lambda: ir_a_imagen())
boton_ir_a_imagen.pack()


# Agrega un Label y un Entry para tolerancia
label_tolerancia1 = tk.Label(marco_botones, text="MV Threshold ON (0.0-0.5):")
label_tolerancia1.pack()

entrada_tolerancia1 = tk.Entry(marco_botones)
entrada_tolerancia1.pack()
entrada_tolerancia1.insert(0, str(tolerancia1))  # Valor por defecto
entrada_tolerancia1.bind('<Return>', lambda event: actualizar_tolerancia1())


# Botón para seleccionar ROI1
boton_seleccionar_roi1 = tk.Button(marco_botones, text="Select ROI1 (static area)", command=lambda: activar_seleccion_roi1())
boton_seleccionar_roi1.pack()
boton_seleccionar_roi1.config(state=tk.DISABLED)  # Deshabilitar inicialmente

boton_calcular_media_roi = tk.Button(marco_botones, text="Calculate averages in ROI1", command=lambda: plot_media_roi())
boton_calcular_media_roi.pack()
boton_calcular_media_roi.config(state=tk.DISABLED)  # Deshabilitar inicialmente

# Agrega un Label y un Entry para tolerancia
label_tolerancia = tk.Label(marco_botones, text="Tolerance (mm):")
label_tolerancia.pack()

entrada_tolerancia = tk.Entry(marco_botones)
entrada_tolerancia.pack()
entrada_tolerancia.insert(0, str(tolerancia))  # Valor por defecto
entrada_tolerancia.bind('<Return>', lambda event: actualizar_tolerancia())



# Botón para seleccionar ROI2 (inicialmente deshabilitado)
boton_seleccionar_roi2 = tk.Button(marco_botones, text="Select ROI2 (moving area)", command=lambda: activar_seleccion_roi2())
boton_seleccionar_roi2.pack()
boton_seleccionar_roi2.config(state=tk.DISABLED)  # Deshabilitar inicialmente


# Botón para calcular correlación entre ROI2 y otras imágenes
boton_calcular_correlacion = tk.Button(marco_botones, text="Calculate Displacement", command=lambda: analizar_correlacion())
boton_calcular_correlacion.pack()
boton_calcular_correlacion.config(state=tk.DISABLED)  # Deshabilitar inicialmente

# Botón para combinar los plots
boton_resultados = tk.Button(marco_botones, text="Get results", command=lambda: resultado_final())
boton_resultados.pack()
boton_resultados.config(state=tk.DISABLED)  # Deshabilitar inicialmente

# Al calcular los resultados, primero actualiza el valor de `tolerancia`
def actualizar_tolerancia():
    global tolerancia
    try:
        tolerancia = float(entrada_tolerancia.get())
    except ValueError:
        messagebox.showwarning("Warning", "Please enter a valid number for the tolerance.")

def actualizar_tolerancia1():
    global tolerancia1
    try:
        tolerancia1 = float(entrada_tolerancia1.get())
    except ValueError:
        messagebox.showwarning("Warning", "Please enter a valid number for the threshold.")

# Function to select the folder with images
def seleccionar_carpeta():
    global ruta_archivos, lista_archivos, indice_actual, delta_ms_values, ms_value
    ruta_archivos = filedialog.askdirectory(title="Select folder with .his files")
    
    if not ruta_archivos:
        return
    
    ruta_actual.set(f"Folder: {ruta_archivos}")  
    
    lista_archivos = [f for f in os.listdir(ruta_archivos) if f.endswith(".his")]
    lista_archivos.sort()  

    if not lista_archivos:
        messagebox.showwarning("Warning", "No .his files were found in the selected folder.")
        boton_seleccionar_roi1.config(state=tk.DISABLED) # Habilitar el botón de seleccionar ROI1  
        boton_calcular_media_roi.config(state=tk.DISABLED)
        boton_seleccionar_roi2.config(state=tk.DISABLED)
        boton_calcular_correlacion.config(state=tk.DISABLED)
        return
    else:
        boton_seleccionar_roi1.config(state=tk.NORMAL) # Habilitar el botón de seleccionar ROI1  
        boton_calcular_media_roi.config(state=tk.DISABLED)  
        boton_seleccionar_roi2.config(state=tk.DISABLED)
        boton_calcular_correlacion.config(state=tk.DISABLED)  
        boton_resultados.config(state=tk.DISABLED) 


    
    indice_actual = 0
    mostrar_imagen(indice_actual)

    archivo_xml = None
    for archivo in os.listdir(ruta_archivos):
        if archivo.endswith(".xml"):
            archivo_xml = os.path.join(ruta_archivos, archivo)
            break

    if not archivo_xml:
        messagebox.showwarning("Warning", "No XML file found in the selected folder.")
        return
    
    try:
        tree = ET.parse(archivo_xml)  
        root = tree.getroot()
        delta_ms_values = []
        for frame in root.findall('.//Frame'):
            delta_ms = frame.find('DeltaMs').text
            delta_ms_values.append(int(delta_ms))
##  ms_value_element = root.find('.//ms') # Encontrar el valor de ms
##        if ms_value_element is not None:
##        # Convertir el valor de ms a entero
##            ms_value = int(ms_value_element.text)
##        else:
##            print('No se encontró el valor de ms.')
##        delta_ms_values = np.arange(0, len(lista_archivos) * ms_value, ms_value)
    except (FileNotFoundError, ET.ParseError):
        messagebox.showerror("Error", "Error loading the XML file.")


# Function to read .his files
def leer_his(ruta_imagen):
    try:
        with open(ruta_imagen, 'rb') as f:
            imagen = 65535 - np.fromfile(f, dtype=np.uint16)
        imagen = imagen[50:].reshape((512, 512))
        return imagen
    except Exception as e:
        messagebox.showerror("Error", f"Error reading the image: {e}")
        return np.zeros((512, 512))

nombre_imagen_actual = tk.StringVar()  
#etiqueta_nombre_imagen = tk.Label(marco_botones, textvariable=nombre_imagen_actual)
#etiqueta_nombre_imagen.pack()
etiqueta_nombre_imagen = tk.Label(ventana_principal, textvariable=nombre_imagen_actual)
etiqueta_nombre_imagen.pack(side=tk.BOTTOM)  # Coloca la etiqueta en la parte inferior de la ventana


def mostrar_imagen(indice):
    global lista_archivos, entrada_numero_imagen, rectangulo_roi1, rectangulo_roi2, roi1, roi2
    if lista_archivos:
        ruta_imagen = os.path.join(ruta_archivos, lista_archivos[indice])
        imagen = leer_his(ruta_imagen)
        
        imagen = (255-imagen // 256).astype(np.uint8)  
        imagen_actual = Image.fromarray(imagen)
        imagen_tk = ImageTk.PhotoImage(imagen_actual)

        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=imagen_tk)

        etiqueta_imagen_actual.image = imagen_tk
        texto_imagen.set(f"Image {indice + 1} of {len(lista_archivos)}")
        nombre_imagen_actual.set(lista_archivos[indice])  
        entrada_numero_imagen.delete(0, tk.END)
        entrada_numero_imagen.insert(0, str(indice + 1))

    if roi1 is not None:
        x1, y1, x2, y2 = roi1
        rectangulo_roi1 = canvas.create_rectangle(x1, y1, x2, y2, outline='green', width=4)
    if roi2 is not None:
        x1, y1, x2, y2 = roi2
        rectangulo_roi2 = canvas.create_rectangle(x1, y1, x2, y2, outline='yellow', width=4, dash=(10, 10))

# Function to calculate the mean of the ROI1 in the current image
def calcular_media_roi1(imagen):
    global roi1
    if roi1 is None:
        messagebox.showwarning("Warning", "No ROI1 has been selected.")
        return None
    x1, y1, x2, y2 = roi1
    roi1_recortada = imagen[y1:y2, x1:x2]
    return np.mean(roi1_recortada)


##def calcular_desplazamiento(roi_actual, imagen_comparativa, roi, rango_busqueda=20):
##    mejor_desplazamiento = (0, 0)
##    x1, y1, x2, y2 = roi
##    menor_ssd = float('inf')
##
##    # Media del ROI actual
##    media_roi_actual = np.mean(roi_actual)
##
##
##    # Definimos el rango de búsqueda para el desplazamiento
##    for dx in range(-rango_busqueda, rango_busqueda + 1):
##        for dy in range(-rango_busqueda, rango_busqueda + 1):
##            # Verificamos que no nos salimos de las dimensiones de roi_comparativa
##            if (0 <= y1 + dy < imagen_comparativa.shape[0] and 
##                0 <= x1 + dx < imagen_comparativa.shape[1] and 
##                0 <= y2 + dy < imagen_comparativa.shape[0] and 
##                0 <= x2 + dx < imagen_comparativa.shape[1]):
##                
##                # Crear ROI desplazada
##                roi_shifted = imagen_comparativa[y1 + dy:y2 + dy, x1 + dx:x2 + dx]
##
##                if roi_shifted.shape[0] != roi_actual.shape[0] or roi_shifted.shape[1] != roi_actual.shape[1]:
##                    continue
##
##                # Media del ROI desplazado
##                media_roi_shifted = np.mean(roi_shifted)
##
##                # Restamos la media para compensar el cambio de brillo
##                ssd = np.sum((roi_actual - media_roi_actual) - (roi_shifted - media_roi_shifted)) ** 2
##
##                if ssd < menor_ssd:
##                    menor_ssd = ssd
##                    mejor_desplazamiento = np.sqrt(dx**2 + dy**2)
##    
##    return mejor_desplazamiento


def calcular_desplazamiento_match_template(roi_actual, roi_comparativa, roi):
    x1, y1, x2, y2 = roi

    # Aplicar la función de búsqueda de plantilla
    result = cv2.matchTemplate(roi_comparativa.astype(np.float32), roi_actual.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    
    # Encuentra la posición del valor máximo
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # max_loc es la esquina superior izquierda del mejor ajuste
    #return np.sqrt((max_loc[0]-x1)**2 + (max_loc[1]-y1)**2)
    return (y1 - max_loc[1])


# Function to calculate the shift of ROI2 with all images
def analizar_correlacion():
    global desplazamientos, desplazamientos_filtered
    global delta_ms_IN_pos, delta_ms_IN_neg, delta_ms_OUT_pos, delta_ms_OUT_neg

    desplazamientos = []
    delta_ms_IN_pos = []
    delta_ms_IN_neg = []
    delta_ms_OUT_pos = []
    delta_ms_OUT_neg = []

    actualizar_tolerancia()

    if roi2 is None:
        messagebox.showwarning("Warning", "No ROI2 has been selected.")
        return

    # ---------------------- ROI2 base ----------------------
    imagen_actual = leer_his(os.path.join(ruta_archivos, lista_archivos[indice_actual]))
    x1, y1, x2, y2 = roi2
    roi2_actual = imagen_actual[y1:y2, x1:x2]

    # ---------------------- desplazamiento ----------------------
    for archivo in lista_archivos:
        imagen = leer_his(os.path.join(ruta_archivos, archivo))
        d_px = calcular_desplazamiento_match_template(roi2_actual, imagen, roi2)
        desplazamientos.append(d_px * escala_pixel)   # ya en mm

    desplazamientos = np.array(desplazamientos)

    # ---------------------- filtrado ----------------------
    offset = tamano_filtro // 2
    desplazamientos_filtered = np.array(
        filtro_mediana(desplazamientos, tamano_filtro)
    )

    # tiempos alineados con señal filtrada
    t_filtered = np.array(delta_ms_values[offset : offset + len(desplazamientos_filtered)])

    # ---------------------- detección IN/OUT ----------------------
    pos_in_pos, pos_in_neg, pos_out_pos, pos_out_neg = detectar_cambio2(
        desplazamientos_filtered,
        tolerancia
    )

    # ---------------------- función auxiliar: interpolación exacta ----------------------
    def tiempo_interpolado(j, valor_objetivo):
        """Devuelve el TIEMPO EXACTO donde la señal filtrada cruza el umbral."""
        if j <= 0 or j >= len(desplazamientos_filtered):
            return t_filtered[j]

        v0 = desplazamientos_filtered[j-1]
        v1 = desplazamientos_filtered[j]
        t0 = t_filtered[j-1]
        t1 = t_filtered[j]

        if v1 == v0:
            return t1

        return t0 + (t1 - t0) * (valor_objetivo - v0) / (v1 - v0)

    # ---------------------- convertir índices en tiempos ----------------------
    for j in pos_in_pos:
        delta_ms_IN_pos.append(tiempo_interpolado(j, +tolerancia))

    for j in pos_in_neg:
        delta_ms_IN_neg.append(tiempo_interpolado(j, -tolerancia))

    for j in pos_out_pos:
        delta_ms_OUT_pos.append(tiempo_interpolado(j, +tolerancia))

    for j in pos_out_neg:
        delta_ms_OUT_neg.append(tiempo_interpolado(j, -tolerancia))

    # ---------------------- GRAFICADO ----------------------
    plt.figure(figsize=(10, 5))

    plt.plot(
        t_filtered,
        desplazamientos_filtered,
        color='red', label='ROI2 Median Filter', linewidth=1.2
    )

    plt.scatter(delta_ms_IN_pos,  [tolerancia]*len(delta_ms_IN_pos),
                color='green', marker='+', s=100, label='Tolerance IN')
    plt.scatter(delta_ms_IN_neg,  [-tolerancia]*len(delta_ms_IN_neg),
                color='green', marker='+', s=100)

    plt.scatter(delta_ms_OUT_pos, [tolerancia]*len(delta_ms_OUT_pos),
                color='red', marker='+', s=100, label='Tolerance OUT')
    plt.scatter(delta_ms_OUT_neg, [-tolerancia]*len(delta_ms_OUT_neg),
                color='red', marker='+', s=100)

    plt.title('ROI2 Motion Tracking and Tolerance Crossings')
    plt.xlabel('Time (ms)')
    plt.ylabel('ROI2 Motion (mm)')
    plt.grid()
    plt.legend()
    plt.show()


    # Clear the ROI after calculation
    #roi2 = None  
    boton_calcular_media_roi.config(state=tk.DISABLED)  
    boton_seleccionar_roi1.config(state=tk.NORMAL)  
    boton_seleccionar_roi2.config(state=tk.NORMAL)
    boton_calcular_correlacion.config(state=tk.DISABLED)  
    boton_resultados.config(state=tk.NORMAL) 

def detectar_cambio(vector_valores, tolerancia=0.1, umbral=0.1):
    # Buscar cambios abruptos
    # deltas = np.abs(np.diff(vector_valores))
    # cambios_abruptos = np.where(deltas > tolerancia)[0]

    # Comparar con umbral
    umbral_estado_minimo = min(vector_valores) + umbral * np.ptp(vector_valores)
    umbral_estado_maximo = min(vector_valores) + (1-umbral) * np.ptp(vector_valores)
    # umbral_estado_maximo = vector_valores.max() - umbral * np.ptp(vector_valores)

    # Combinar técnicas
    estados = np.where(vector_valores > umbral_estado_minimo, 1, 0)
    estados2 = np.where(vector_valores > umbral_estado_maximo, 1, 0)

    # Identificar cambios
    #cambios = np.where(np.diff(estados) == 1)[0]
    # Definir las listas para almacenar las posiciones de los valores 1 y las posiciones de los valores -1
    posiciones_ON = np.where(np.diff(estados) == 1)[0] + 1
    posiciones_OFF = np.where(np.diff(estados2) == -1)[0] + 1

    return umbral_estado_minimo, umbral_estado_maximo, posiciones_ON, posiciones_OFF

def detectar_cambio2(vector_valores, tolerancia):

    estados = np.where(np.abs(vector_valores) > tolerancia, 1, 0)

    # Identificar cambios
    #cambios = np.where(np.diff(estados) == 1)[0]
    # Definir las listas para almacenar las posiciones de los valores 1 y las posiciones de los valores -1
    posiciones_ON = np.where(np.diff(estados) == -1)[0] + 1
    posiciones_OFF = np.where(np.diff(estados) == 1)[0] + 1

    # Separar posiciones ON y OFF en positivas y negativas
    posiciones_ON_positivas = []
    posiciones_ON_negativas = []
    posiciones_OFF_positivas = []
    posiciones_OFF_negativas = []

    for pos in posiciones_ON:
        if vector_valores[pos-1] > 0:
            posiciones_ON_positivas.append(pos)
        else:
            posiciones_ON_negativas.append(pos)

    for pos in posiciones_OFF:
        if vector_valores[pos] > 0:
            posiciones_OFF_positivas.append(pos)
        else:
            posiciones_OFF_negativas.append(pos)

    return (posiciones_ON_positivas, posiciones_ON_negativas, 
            posiciones_OFF_positivas, posiciones_OFF_negativas)

def filtro_mediana(datos, tamano_filtro):
    medias = []
    
    for i in range(tamano_filtro // 2, len(datos) - tamano_filtro // 2):
        sublista = datos[i - tamano_filtro // 2  : i + tamano_filtro // 2 + 1]
        medias.append(np.median(sublista))
    
    return medias

# Function to plot the mean of the ROIs
def plot_media_roi():
    global tolerancia1, valores_media_roi1, valores_media_roi1_filtered, delta_ms_ON, delta_ms_OFF, umbral_estado_minimo, umbral_estado_maximo

    valores_media_roi1 = []
    actualizar_tolerancia1()  # Actualiza el valor de tolerancia antes de la correlación

    boton_seleccionar_roi1.config(state=tk.DISABLED)
    boton_calcular_media_roi.config(state=tk.DISABLED)  
    boton_seleccionar_roi2.config(state=tk.DISABLED)
    boton_calcular_correlacion.config(state=tk.DISABLED)  # Habilitar el botón de calcular correlación
    boton_resultados.config(state=tk.DISABLED)

    delta_ms_ON = []
    delta_ms_OFF = []
    
    for indice, archivo in enumerate(lista_archivos):
        ruta_imagen = os.path.join(ruta_archivos, archivo)
        imagen = leer_his(ruta_imagen)

        media_roi1 = calcular_media_roi1(imagen)
        if media_roi1 is not None:
            valores_media_roi1.append(media_roi1)

    if not valores_media_roi1:
        messagebox.showwarning("Warning", "The mean value could not be calculated for one or more ROIs.")
        return

    valores_media_roi1_filtered = filtro_mediana(np.array(valores_media_roi1), tamano_filtro)
    #### aquí se puede cambiar el valor de la tolerancia (valor entre 0 y 1)
    umbral_estado_minimo, umbral_estado_maximo, posiciones_ON, posiciones_OFF = detectar_cambio(valores_media_roi1_filtered, 0.1, tolerancia1)





    for i in posiciones_ON:
        m = ( valores_media_roi1_filtered[i] - valores_media_roi1_filtered[i-1] ) / ( delta_ms_values[i + tamano_filtro // 2] - delta_ms_values[i - 1 + tamano_filtro // 2])
        delta_ms_ON.append( (umbral_estado_minimo - valores_media_roi1_filtered[i]) / m + delta_ms_values[i + tamano_filtro // 2])
        
    for i in posiciones_OFF:
        m = ( valores_media_roi1_filtered[i] - valores_media_roi1_filtered[i-1] ) / ( delta_ms_values[i + tamano_filtro // 2] - delta_ms_values[i - 1 + tamano_filtro // 2])
        delta_ms_OFF.append( (umbral_estado_maximo - valores_media_roi1_filtered[i]) / m + delta_ms_values[i + tamano_filtro // 2] )

    ### delta_ms_ON y delta_ms_OFF contienen los tiempos en los que se produce el arranque del haz de MV y el apagado por el SGRT
 

    plt.figure(figsize=(10, 5))
    plt.plot(delta_ms_values[:len(valores_media_roi1)], valores_media_roi1, color='lightblue', label='ROI1 Mean', linewidth=2.0)
    plt.plot(delta_ms_values[tamano_filtro // 2 :len(delta_ms_values) - tamano_filtro// 2 ], valores_media_roi1_filtered, color='black', label='Median Filter', linewidth=0.5)
    # Graficar todos los puntos en azul
    #plt.scatter(delta_ms_values, valores_media_roi1, color='blue', label='Todas las medidas')

    # Graficar los puntos en las posiciones deseadas
    # Se asume que 'posiciones' contiene índices válidos para 'valores_media_roi1'
    # for i in pos1_ON:
        

    # Agregar los puntos al gráfico
    plt.scatter(delta_ms_ON, np.full(len(delta_ms_ON), umbral_estado_minimo), color='darkgreen', marker='x', s=100, label='Beam ON')

    # Graficar los puntos en las posiciones deseadas
    # Se asume que 'posiciones' contiene índices válidos para 'valores_media_roi1'

    # Agregar los puntos al gráfico
    plt.scatter(delta_ms_OFF, np.full(len(delta_ms_OFF), umbral_estado_maximo), color='red', marker='x', s=100, label='Beam OFF')


    plt.title('ROI1 Noise Analysis and Beam ON/OFF Detection')
    plt.xlabel('Time (ms)')
    plt.ylabel('Mean Pixel Value (ROI1)')
    
    plt.xticks(np.arange(0, 1.1 * max(delta_ms_values), 10000))
    plt.grid()
    plt.legend()
    plt.show()

    # Clear the ROI after calculation
    #roi1 = None  
    boton_seleccionar_roi1.config(state=tk.NORMAL)  
    boton_seleccionar_roi2.config(state=tk.NORMAL)  

# Function to show the previous image
def mostrar_anterior():
    global indice_actual
    if indice_actual > 0:
        indice_actual -= 1
        mostrar_imagen(indice_actual)

# Function to show the next image
def mostrar_siguiente():
    global indice_actual
    if indice_actual < len(lista_archivos) - 1:
        indice_actual += 1
        mostrar_imagen(indice_actual)

# Function to go to the specified image in the entry
def ir_a_imagen():
    global indice_actual
    try:
        indice_nuevo = int(entrada_numero_imagen.get()) - 1
        if 0 <= indice_nuevo < len(lista_archivos):
            indice_actual = indice_nuevo
            mostrar_imagen(indice_actual)
        else:
            messagebox.showwarning("Warning", "Index out of range.")
    except ValueError:
        messagebox.showwarning("Warning", "Please enter a valid number.")

# Variables for the ROI and rectangle
rectangulo_roi1 = None
rectangulo_roi2 = None
inicio_x, inicio_y = None, None
inicio_x_roi2, inicio_y_roi2 = None, None
seleccionando_roi1 = False
seleccionando_roi2 = False

def activar_seleccion_roi1():
    global seleccionando_roi1, boton_seleccionar_roi1, seleccionando_roi1, boton_seleccionar_roi2
    seleccionando_roi1 = not seleccionando_roi1  

    boton_calcular_media_roi.config(state=tk.DISABLED)  
    boton_seleccionar_roi2.config(state=tk.DISABLED)
    boton_calcular_correlacion.config(state=tk.DISABLED)  # Habilitar el botón de calcular correlación
    boton_resultados.config(state=tk.DISABLED)


    if seleccionando_roi1:
        print("Selecting ROI1 (stationary region) activated.")
        borrar_rectangulo_roi1()  
    else:
        print("Selecting ROI1 (stationary region) deactivated.")
        if roi1 is not None:
            boton_calcular_media_roi.config(state=tk.NORMAL)  
            boton_seleccionar_roi1.config(state=tk.DISABLED)  

def borrar_rectangulo_roi1():
    global rectangulo_roi1
    if rectangulo_roi1:  
        canvas.delete(rectangulo_roi1)
        rectangulo_roi1 = None



def on_click(event):
    global inicio_x, inicio_y, rectangulo_roi1, inicio_x_roi2, inicio_y_roi2, rectangulo_roi2
    if not seleccionando_roi1 and not seleccionando_roi2:
        return

    if seleccionando_roi1:
        inicio_x = event.x
        inicio_y = event.y
        borrar_rectangulo_roi1()  
        rectangulo_roi1 = canvas.create_rectangle(inicio_x, inicio_y, inicio_x, inicio_y, outline='green', width=4)
    elif seleccionando_roi2:
        inicio_x_roi2 = event.x
        inicio_y_roi2 = event.y
        borrar_rectangulo_roi2()  
        rectangulo_roi2 = canvas.create_rectangle(inicio_x_roi2, inicio_y_roi2, inicio_x_roi2, inicio_y_roi2, outline='yellow', width=4, dash=(10, 10))

def on_drag(event):
    global inicio_x, inicio_y, rectangulo_roi1, inicio_x_roi2, inicio_y_roi2, rectangulo_roi2
    if rectangulo_roi1 and seleccionando_roi1:
        canvas.coords(rectangulo_roi1, inicio_x, inicio_y, event.x, event.y)
    elif rectangulo_roi2 and seleccionando_roi2:
        canvas.coords(rectangulo_roi2, inicio_x_roi2, inicio_y_roi2, event.x, event.y)

def on_release(event):
    global roi1, roi2  
    if not seleccionando_roi1 and not seleccionando_roi2:
        return
    if seleccionando_roi1:
        fin_x = event.x
        fin_y = event.y
        roi1 = (min(inicio_x, fin_x), min(inicio_y, fin_y), max(inicio_x, fin_x), max(inicio_y, fin_y))
        print(f'ROI1 selected: {roi1}')
    elif seleccionando_roi2:
        fin_x = event.x
        fin_y = event.y
        roi2 = (min(inicio_x_roi2, fin_x), min(inicio_y_roi2, fin_y), max(inicio_x_roi2, fin_x), max(inicio_y_roi2, fin_y))
        print(f'ROI2 selected: {roi2}')
            

def activar_seleccion_roi2():
    global seleccionando_roi2, boton_seleccionar_roi2, boton_seleccionar_roi1, boton_calcular_correlacion
    seleccionando_roi2 = not seleccionando_roi2  

    boton_seleccionar_roi1.config(state=tk.DISABLED)
    boton_calcular_media_roi.config(state=tk.DISABLED)  
    boton_calcular_correlacion.config(state=tk.DISABLED)  # Habilitar el botón de calcular correlación
    boton_resultados.config(state=tk.DISABLED)

    if seleccionando_roi2:
        print("Selecting ROI2 activated.")
        borrar_rectangulo_roi2()  
    else:
        print("Selecting ROI2 deactivated.")
        if roi2 is not None:
            boton_calcular_correlacion.config(state=tk.NORMAL)  # Habilitar el botón de calcular correlación
            boton_seleccionar_roi2.config(state=tk.NORMAL)  

def borrar_rectangulo_roi2():
    global rectangulo_roi2
    if rectangulo_roi2:  
        canvas.delete(rectangulo_roi2)
        rectangulo_roi2 = None  

# Función para manejar la selección
def onselect(eclick, erelease, ax1):
    global labels  # Accedemos a la lista global de etiquetas
    # Borrar etiquetas previas
    for label in labels:
        label.remove()
    labels = []  # Reiniciar la lista de etiquetas
    
    # Obtener los límites del intervalo seleccionado
    x1, x2 = eclick.xdata, erelease.xdata


    # Convertir listas a arrays de NumPy y crear máscaras booleanas

    if delta_ms_IN_pos == []:
        delta_ms_IN_array = np.array(delta_ms_IN_neg)
    elif delta_ms_IN_neg == []:
        delta_ms_IN_array = np.array(delta_ms_IN_pos)
    else:
        delta_ms_IN_array = np.sort(np.concatenate(np.array(delta_ms_IN_pos), np.array(delta_ms_IN_pos)))
        
    if delta_ms_OUT_pos == []:
        delta_ms_OUT_array = np.array(delta_ms_OUT_neg)
    elif delta_ms_OUT_neg == []:
        delta_ms_OUT_array = np.array(delta_ms_OUT_pos)
    else:
        delta_ms_OUT_array = np.sort(np.concatenate(np.array(delta_ms_OUT_pos), np.array(delta_ms_OUT_pos)))
        
    delta_ms_ON_array = np.array(delta_ms_ON)
    delta_ms_OFF_array = np.array(delta_ms_OFF)

    # Filtrar los datos para encontrar puntos relevantes dentro del intervalo
    in_mask = (delta_ms_IN_array >= x1) & (delta_ms_IN_array <= x2)
    on_mask = (delta_ms_ON_array >= x1) & (delta_ms_ON_array <= x2)
    out_mask = (delta_ms_OUT_array >= x1) & (delta_ms_OUT_array <= x2)
    off_mask = (delta_ms_OFF_array >= x1) & (delta_ms_OFF_array <= x2)

        # Obtener los tiempos dentro del intervalo usando la máscara
    in_times = delta_ms_IN_array[in_mask]
    on_times = delta_ms_ON_array[on_mask]
    out_times = delta_ms_OUT_array[out_mask]
    off_times = delta_ms_OFF_array[off_mask]

    diffs_in_on = []
    diffs_out_off = []

    # Comprobar que ambos arrays tengan la misma longitud
    min_len_in_on = min(len(in_times), len(on_times))
    min_len_out_off = min(len(out_times), len(off_times))

    # Calcular diferencias ON - IN (pareja a pareja)
    if min_len_in_on > 0:
        diffs_in_on = [on_times[i] - in_times[i] for i in range(min_len_in_on)]
        diffs_in_on_pos = [d for d in diffs_in_on if d > 0]
        average_in_on = np.mean(diffs_in_on_pos)
        std_in_on = np.std(diffs_in_on_pos)
        n_in_on = len(diffs_in_on_pos)

        # Si cualquiera es NaN, no pintamos nada
        if not (np.isnan(average_in_on) or np.isnan(std_in_on)):
            print(f"Differences ON - IN mean: {average_in_on:.2f} (σ={std_in_on:.2f})")
            # Agregar etiqueta para el promedio ON - IN
            label_in_on = ax1.text(x=ax1.get_xlim()[0]*1 , y=ax1.get_ylim()[0]*1.0, 
                                    s=f'Turn‑On Latency (n={n_in_on:.0f}): {average_in_on:.0f} (σ={std_in_on:.2f}) ms', 
                                    fontsize=10, color='darkgreen', verticalalignment='bottom', 
                                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='darkgreen'))
            labels.append(label_in_on)
    else:
        print("No data available to calculate ON - IN differences.")

    # Calcular diferencias OFF - OUT (pareja a pareja)
    if min_len_out_off > 0:
        diffs_out_off = [off_times[i] - out_times[i] for i in range(min_len_out_off)]
        diffs_out_off_pos = [d for d in diffs_out_off if d > 0]
        average_out_off = np.mean(diffs_out_off_pos)
        std_out_off = np.std(diffs_out_off_pos)
        n_out_off = len(diffs_out_off_pos)

        # Si cualquiera es NaN, no pintamos nada
        if not (np.isnan(average_out_off) or np.isnan(std_out_off)):
            print(f"Differences OFF - OUT mean: {average_out_off:.2f} (σ={std_out_off:.2f})")
           # Agregar etiqueta para el promedio OFF - OUT
            label_out_off = ax1.text(x=(ax1.get_xlim()[1]+ax1.get_xlim()[0])*0.5, y=ax1.get_ylim()[0]*1.0, 
                                      s=f'Turn‑Off Latency (n={n_out_off:.0f}): {average_out_off:.0f} (σ={std_out_off:.2f}) ms', 
                                      fontsize=10, color='red', verticalalignment='bottom', 
                                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
            labels.append(label_out_off)
    else:
        print("No data available to calculate OFF - OUT differences.")

    # Redibujar el gráfico con las nuevas etiquetas
    plt.draw()
    






# -*- coding: utf-8 -*-
# Archivo generado automáticamente: resultado_final_function.py
# Contiene una versión limpia y funcional de la función `resultado_final()`
# para reemplazar la versión corrupta en tu script principal.


def resultado_final():
    global valores_media_roi1, valores_media_roi1_filtered
    global delta_ms_ON, delta_ms_OFF
    global desplazamientos_filtered
    global delta_ms_IN_pos, delta_ms_IN_neg, delta_ms_OUT_pos, delta_ms_OUT_neg

    # Asegurar tolerancia actualizada desde la UI
    actualizar_tolerancia()

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector

    # Asegurar arrays
    valores_media_roi1_filtered = np.array(valores_media_roi1_filtered)
    desplazamientos_filtered = np.array(desplazamientos_filtered)

    # Bloquear UI durante el cálculo
    boton_seleccionar_roi1.config(state='disabled')
    boton_calcular_media_roi.config(state='disabled')
    boton_seleccionar_roi2.config(state='disabled')
    boton_calcular_correlacion.config(state='disabled')
    boton_resultados.config(state='disabled')

    offset = tamano_filtro // 2

    # ===============================
    # ROI1 (media de píxeles)
    # ===============================
    t1_filt = np.array(delta_ms_values[offset: offset + len(valores_media_roi1_filtered)])

    def indice_mas_cercano(t, arr):
        return int(np.argmin(np.abs(arr - t)))

    def tiempo_interpolado_roi1(j, valor_objetivo):
        # Interpola el tiempo exacto donde ROI1 filtrada cruza valor_objetivo
        if j <= 0 or j >= len(valores_media_roi1_filtered):
            return t1_filt[j]
        v0 = valores_media_roi1_filtered[j-1]
        v1 = valores_media_roi1_filtered[j]
        t0 = t1_filt[j-1]
        t1 = t1_filt[j]
        if v1 == v0:
            return t1
        return t0 + (t1 - t0) * (valor_objetivo - v0) / (v1 - v0)

    # Proyecta ON/OFF fuera de rango a los extremos del eje filtrado
    def tiempo_roi1_en_rango(t_ev):
        if t_ev <= t1_filt[0]:
            return tiempo_interpolado_roi1(0, umbral_estado_minimo)
        if t_ev >= t1_filt[-1]:
            return tiempo_interpolado_roi1(len(t1_filt) - 1, umbral_estado_minimo)
        j = indice_mas_cercano(t_ev, t1_filt)
        return tiempo_interpolado_roi1(j, umbral_estado_minimo)

    tiempos_ON  = [tiempo_roi1_en_rango(t) for t in delta_ms_ON]
    tiempos_OFF = [tiempo_roi1_en_rango(t) for t in delta_ms_OFF]

    fig, ax1 = plt.subplots()

    ax1.plot(np.array(delta_ms_values)[:len(valores_media_roi1)],
             valores_media_roi1,
             color="lightblue", linewidth=2.0, label="ROI1 Mean")

    ax1.plot(t1_filt, valores_media_roi1_filtered,
             color="black", linewidth=0.5, label="Median Filter")

    ax1.scatter(tiempos_ON,  [umbral_estado_minimo]*len(tiempos_ON),
                color="darkgreen", marker="x", s=100, label="Beam ON")
    ax1.scatter(tiempos_OFF, [umbral_estado_minimo]*len(tiempos_OFF),
                color="red", marker="x", s=100, label="Beam OFF")
##    ax1.scatter(tiempos_OFF, [umbral_estado_maximo]*len(tiempos_OFF),
##                color="red", marker="x", s=100, label="Beam OFF")

    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Mean Pixel Value (ROI1)")
    ax1.grid()
    ax1.legend(loc="upper left")

    # ===============================
    # ROI2: desplazamientos filtrados
    # ===============================
    t2_filt = np.array(delta_ms_values[offset: offset + len(desplazamientos_filtered)])

    ax2 = ax1.twinx()
    ax2.plot(t2_filt, desplazamientos_filtered,
             color="red", linewidth=1.0, label="ROI2 Median Filter")


    # ===============================
    # Escalado estilo "facción" (alineación exacta)
    # ===============================
    min1   = float(valores_media_roi1_filtered.min())
    rango1 = float(np.ptp(valores_media_roi1_filtered))
    min2   = float(desplazamientos_filtered.min())
    rango2 = float(np.ptp(desplazamientos_filtered))
    if rango2 == 0:
        rango2 = 1e-9  # evitar división por cero

    # 1) ROI1: fija los límites con tu margen habitual
    MARGEN_ROI1 = 0.10  # 10% de margen; puedes bajarlo si quieres más compacto
    y1_min = min1 - MARGEN_ROI1*rango1
    y1_max = min1 + (1 + MARGEN_ROI1)*rango1
    ax1.set_ylim(y1_min, y1_max)

    # 2) Fracción vertical de ROI1 donde está el umbral (invariante a tolerancia1)
    f1 = (umbral_estado_minimo - y1_min) / (y1_max - y1_min)  # 0..1

    # 3) Elige "altura visual" de ROI2 en mm (cuán alto/compacto ves el panel ROI2)
    #    Puedes expresar s_mm a partir de tus referencias originales para conservar
    #    el "look" del código viejo. Por ejemplo:
    F_REF1    = 0.0833
    F_REF2    = 0.125
    ALTO_ROI2 = 3.0   # baja a 0.90 si lo quieres más compacto

    ylim_scaling = (MARGEN_ROI1 * rango1) / F_REF1 / ((rango2 - tolerancia) / F_REF2)
    s_mm = ALTO_ROI2 * rango1 / ylim_scaling
    #  -> si lo prefieres aún más directo, puedes usar un s_mm fijo, p.ej.: s_mm = 2.5  # mm

    # 4) ROI2: impón que y=-tolerancia caiga en la fracción f1 del eje (alineación exacta)
    y2_min = -tolerancia - f1*s_mm
    y2_max = y2_min + s_mm
    ax2.set_ylim(y2_min, y2_max)
    ax2.set_ylabel("ROI2 Motion (mm)")

    # 5) Tolerance en mm REALES (sin A*y+B)
    tol_in_visual  = +tolerancia
    tol_out_visual = -tolerancia

    # ===============================
    # TOLERANCE: ya son tiempos (ms) desde analizar_correlacion()
    # ===============================
    tiempos_IN_pos  = np.array(delta_ms_IN_pos,  dtype=float)
    tiempos_IN_neg  = np.array(delta_ms_IN_neg,  dtype=float)
    tiempos_OUT_pos = np.array(delta_ms_OUT_pos, dtype=float)
    tiempos_OUT_neg = np.array(delta_ms_OUT_neg, dtype=float)

    # Limitar X al rango visible para no estirar el eje
    x_min = float(min(t1_filt[0], t2_filt[0])) if len(t2_filt) else float(t1_filt[0])
    x_max = float(max(t1_filt[-1], t2_filt[-1])) if len(t2_filt) else float(t1_filt[-1])

    def clamp_times(arr):
        return arr[(arr >= x_min) & (arr <= x_max)] if arr.size else arr

    tiempos_IN_pos  = clamp_times(tiempos_IN_pos)
    tiempos_IN_neg  = clamp_times(tiempos_IN_neg)
    tiempos_OUT_pos = clamp_times(tiempos_OUT_pos)
    tiempos_OUT_neg = clamp_times(tiempos_OUT_neg)

    # Dibujar puntos TOLERANCE alineados a la escala visual de ROI2
    ax2.scatter(tiempos_IN_pos,  [tol_in_visual]  * len(tiempos_IN_pos),
                color="darkgreen", marker="+", s=100, label="Tolerance IN")
    ax2.scatter(tiempos_IN_neg,  [tol_out_visual] * len(tiempos_IN_neg),
                color="darkgreen", marker="+", s=100)
    ax2.scatter(tiempos_OUT_pos, [tol_in_visual]  * len(tiempos_OUT_pos),
                color="red", marker="+", s=100, label="Tolerance OUT")
    ax2.scatter(tiempos_OUT_neg, [tol_out_visual] * len(tiempos_OUT_neg),
                color="red", marker="+", s=100)

    # Bloquear auto-estirado del eje X
    ax1.set_xlim(x_min, x_max)

    ax2.legend(loc="upper right")

    # Selector de intervalo (tu callback `onselect` calcula latencias en pantalla)
    rectangle_selector = RectangleSelector(
        ax1,
        lambda eclick, erelease: onselect(eclick, erelease, ax1),
        useblit=True, button=[1],
        minspanx=5, minspany=5,
        spancoords="pixels", interactive=True
    )

    plt.title("Gating Signal Evaluation")
    plt.show()




    # Restaurar UI
    boton_seleccionar_roi1.config(state='normal')
    boton_seleccionar_roi2.config(state='normal')
    boton_resultados.config(state='normal')



    

# Canvas for user interaction
canvas = tk.Canvas(ventana_principal, width=512, height=512)
canvas.pack(side=tk.RIGHT)
canvas.bind("<Button-1>", on_click)  
canvas.bind("<B1-Motion>", on_drag)  
canvas.bind("<ButtonRelease-1>", on_release)  
 

ventana_principal.mainloop()
