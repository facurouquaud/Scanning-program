# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:19:17 2026

@author: Luis1
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
from PIL import Image
from scipy.signal import find_peaks
import scan_datafile as sd
#rom skimage.measure import profile_line
import os
from matplotlib.widgets import RectangleSelector
from skimage.measure import profile_line
from scipy.ndimage import map_coordinates


plt.rcParams["font.family"] = "arial"


def graficar(imagen, pixel_size_um, min_fotones, max_fotones):
    """
    Muestra la imagen directamente, calculando los ejes físicos automáticamente.
    
    Parámetros
    ----------
    imagen : 2D array
        Matriz de intensidades (e.g., número de fotones por píxel).
    pixel_size_um : float, opcional
        Tamaño físico de cada píxel en µm (por defecto 1 µm/píxel).
    titulo : str, opcional
        Título que se muestra en la figura.
    """
    # Calcular ejes físicos
    imagen = imagen.T
    nx, ny = imagen.shape
    x_extent = nx * pixel_size_um
    y_extent = ny * pixel_size_um

    fig, ax = plt.subplots(constrained_layout=True)
    im = ax.imshow(
    imagen,
    #cmap='gist_heat',
    cmap = "viridis",
    extent=[0, x_extent, 0, y_extent],
    origin='lower',
    vmin=min_fotones,
    vmax=max_fotones
)

    ax.set_xlabel("x [µm]", fontsize=20)
    ax.set_ylabel("y [µm]", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.colorbar(im, ax=ax, label="Número de fotones")
    ax.set_aspect('equal', adjustable='box')
    plt.show()



class SBRSelector:
    def graficar(imagen, pixel_size_um, min_fotones, max_fotones):
        """
        Muestra la imagen directamente, calculando los ejes físicos automáticamente.
        
        Parámetros
        ----------
        imagen : 2D array
            Matriz de intensidades (e.g., número de fotones por píxel).
        pixel_size_um : float, opcional
            Tamaño físico de cada píxel en µm (por defecto 1 µm/píxel).
        titulo : str, opcional
            Título que se muestra en la figura.
        """
        # Calcular ejes físicos
        imagen = imagen.T
        nx, ny = imagen.shape
        x_extent = nx * pixel_size_um
        y_extent = ny * pixel_size_um

        fig, ax = plt.subplots(constrained_layout=True)
        im = ax.imshow(
        imagen,
        cmap='inferno',
        extent=[0, x_extent, 0, y_extent],
        origin='lower',
        vmin=min_fotones,
        vmax=max_fotones
    )

        ax.set_xlabel("x [µm]", fontsize=20)
        ax.set_ylabel("y [µm]", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.colorbar(im, ax=ax, label="Número de fotones")
        ax.set_aspect('equal', adjustable='box')
        plt.show()
    def __init__(self, imagen):
        self.imagen = imagen

        self.signal_roi = None
        self.background_roi = None

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(imagen.T, cmap="inferno", origin="lower")

        self.ax.set_title(
            "Seleccione ROI de señal y presione Enter"
        )

        self.selector = RectangleSelector(
            self.ax,
            self.on_select,
            useblit=True,
            button=[1],
            interactive=True
        )

        self.mode = "signal"

        self.fig.canvas.mpl_connect(
            "key_press_event",
            self.on_key
        )

        plt.show()

    def on_select(self, eclick, erelease):

        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])

        roi = self.imagen[xmin:xmax, ymin:ymax]

        if self.mode == "signal":
            self.signal_roi = roi
            print("ROI de señal guardada.")
            print("Presione Enter para seleccionar el fondo.")

        elif self.mode == "background":
            self.background_roi = roi
            self.compute_sbr()

    def on_key(self, event):

        if event.key == "enter":

            if self.mode == "signal":
                self.mode = "background"

                self.ax.set_title(
                    "Seleccione ROI de fondo"
                )
                self.fig.canvas.draw()

            elif self.mode == "background":
                pass

    def compute_sbr(self):

        signal_mean = np.mean(self.signal_roi)
        background_mean = np.mean(self.background_roi)

        sbr = signal_mean / background_mean
        sbr_neto = (
            signal_mean - background_mean
        ) / background_mean

        print("\nResultados")
        print(f"Señal media     = {signal_mean:.2f}")
        print(f"Fondo medio     = {background_mean:.2f}")
        print(f"SBR             = {sbr:.2f}")
        print(f"SBR neto        = {sbr_neto:.2f}")

        plt.close(self.fig)

class ROITraceSelector:
    """
    Construye una pila temporal correctamente aun si los datos vienen como:
      datos[i][0] -> frame i
    o formatos más comunes. Selecciona ROI en un frame y calcula la traza
    sobre todos los frames detectados.
    Parámetros:
      data: lista/ndarray con los datos (varios formatos soportados)
      channel_in_elem: índice dentro de cada elemento para elegir canal/frame (por defecto 0)
      display_frame: frame a mostrar para seleccionar (por defecto 0)
    """
    def __init__(self, data, channel_in_elem=0, display_frame=0):
        self.raw = data
        arr = np.asarray(data)

        # Caso específico: lista/tuple donde cada elemento contiene subelementos y
        # datos[i][channel_in_elem] es una imagen 2D -> construir stack temporal.
        stack = None
        try:
            # comprobar si data[i][channel_in_elem] existe y es 2D
            first = data[0].T
            if hasattr(first, "__len__") and len(first) > channel_in_elem:
                test_img = np.asarray(first[channel_in_elem])
                if test_img.ndim == 2:
                    # construir stack de frames: datos[0][channel_in_elem], datos[1][channel_in_elem], ...
                    frames = []
                    for i, elem in enumerate(data):
                        try:
                            img = np.asarray(elem[channel_in_elem])
                        except Exception:
                            raise ValueError(f"No pude acceder a data[{i}][{channel_in_elem}]")
                        if img.ndim != 2:
                            raise ValueError(f"Esperaba imagen 2D en data[{i}][{channel_in_elem}], obtuve ndim={img.ndim}")
                        frames.append(img)
                    stack = np.stack(frames, axis=0)  # (n_frames, rows, cols)
        except Exception:
            stack = None

        # Si no se construyó, fallback a heurísticas previas (ndarray 3D/4D)
        if stack is None:
            if arr.ndim == 3:
                stack = arr  # (frames, rows, cols)
            elif arr.ndim == 4:
                s0, s1 = arr.shape[0], arr.shape[1]
                if s0 <= 4 and s1 > 4:
                    stack = arr[channel_in_elem]  # (frames, rows, cols)
                elif s1 <= 4 and s0 > 4:
                    stack = arr[:, channel_in_elem]
                else:
                    # intentar extraer último eje como canal
                    if arr.shape[-1] <= 4 and arr.shape[0] > arr.shape[-1]:
                        stack = arr[..., channel_in_elem].copy()
                    else:
                        stack = arr[:, 0]
            else:
                raise ValueError(f"Formato no soportado: ndim={arr.ndim}, shape={arr.shape}")

        if stack.ndim != 3:
            raise ValueError(f"Después de normalizar, stack debe ser 3D, obtenido {stack.shape}")

        self.stack = stack
        self.n_frames, self.n_rows, self.n_cols = stack.shape

        if not (0 <= display_frame < self.n_frames):
            raise ValueError("display_frame fuera de rango")
        self.display_frame = display_frame

        self.roi_slices = None
        self.trace = None

        # Mostrar frame para seleccionar ROI
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.stack[self.display_frame], cmap="inferno", origin="lower")
        self.ax.set_title("Seleccione ROI y presione Enter")
        self.selector = RectangleSelector(self.ax, self.on_select,
                                          useblit=True, button=[1], interactive=True)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        plt.show()

    def on_select(self, eclick, erelease):
        x1, y1 = int(round(eclick.xdata)), int(round(eclick.ydata))
        x2, y2 = int(round(erelease.xdata)), int(round(erelease.ydata))
        col_min, col_max = sorted([x1, x2])
        row_min, row_max = sorted([y1, y2])

        if col_min == col_max:
            col_min = max(0, col_min - 1); col_max = min(self.n_cols, col_max + 1)
        if row_min == row_max:
            row_min = max(0, row_min - 1); row_max = min(self.n_rows, row_max + 1)

        row_min = max(0, min(row_min, self.n_rows - 1))
        row_max = max(0, min(row_max, self.n_rows))
        col_min = max(0, min(col_min, self.n_cols - 1))
        col_max = max(0, min(col_max, self.n_cols))

        if row_min >= row_max or col_min >= col_max:
            print("ROI vacía o inválida.")
            return

        self.roi_slices = (row_min, row_max, col_min, col_max)
        print(f"ROI guardada: filas {row_min}:{row_max}, cols {col_min}:{col_max}. Presione Enter.")

    def on_key(self, event):
        if event.key == "enter" and self.roi_slices is not None:
            self.compute_and_plot_trace()

    def compute_and_plot_trace(self):
        r0, r1, c0, c1 = self.roi_slices
        roi_stack = self.stack[:, r0:r1, c0:c1]
        if roi_stack.size == 0:
            print("ROI vacía — no se puede calcular traza.")
            return
        trace = roi_stack.reshape(self.n_frames, -1).sum(axis=1)
        if np.any(np.isnan(trace)):
            print("Advertencia: la traza contiene NaN.")
        fig2, ax2 = plt.subplots()
        ax2.plot(np.arange(self.n_frames), trace, marker='o', color = "firebrick")
        ax2.set_xlabel("Frame", fontsize = 14); ax2.set_ylabel("Número de fotones", fontsize = 14)
        ax2.grid(True)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        plt.show()
        self.trace = trace
        return trace


def line_profile_interactive(image, pixel_size_um=1.0, linewidth=1, mode='nearest'):
    """
    Selecciona dos puntos con el ratón (clic izquierdo) y devuelve/traza el perfil.
    image: 2D numpy array (no transponer aquí).
    pixel_size_um: tamaño de píxel para eje x físico.
    linewidth: ancho en píxeles para promediar perpendicularmente (int).
    mode: 'nearest'/'constant' para profile_line.
    """
    fig, ax = plt.subplots()
    ax.imshow(image.T, origin='lower', cmap='inferno')
    ax.set_title("Clic en dos puntos para definir la línea")
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)
    if image.ndim != 2:
        raise ValueError("image debe ser 2D")

    fig, ax = plt.subplots()
    ax.imshow(image.T, origin='lower', cmap='inferno')
    ax.set_title("Clic en dos puntos para definir la línea")
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)
    
    if len(pts) < 2:
        print("No se seleccionaron 2 puntos.")
        return None
    
    # Convertir coordenadas de visualización (x,y) a índices (row, col) en image
    (x0, y0), (x1, y1) = pts
    p0 = (float(y0), float(x0))
    p1 = (float(y1), float(x1))
    
    # Longitud en píxeles y muestreo
    length = int(np.hypot(p1[0] - p0[0], p1[1] - p0[1])) + 1
    if length < 2:
        print("La línea es demasiado corta.")
        return None
    
    rr = np.linspace(p0[0], p1[0], length)
    cc = np.linspace(p0[1], p1[1], length)
    
    try:
        prof = map_coordinates(image, [rr, cc], order=1, mode='nearest')
    except Exception as e:
        print("Error al interpolar el perfil:", e)
        return None
    
    x_phys = np.linspace(0, (len(prof) - 1) * pixel_size_um, len(prof))
    
    fig2, ax2 = plt.subplots()
    ax2.plot(x_phys, prof, marker='o', color='firebrick')
    ax2.set_xlabel('Distancia [µm]')
    ax2.set_ylabel('Intensidad (fotones)')
    ax2.grid(True)
    plt.show()
    
    return x_phys, prof
    
    
    





 #%%   
# folder_path = r"C:\Users\Luis1\Downloads\Bungarotoxinas\Comparacion fijadores\Glyoxal"
# file_name = "6x6.png"
# full_path = os.path.join(folder_path, file_name)

path = r"C:\Users\Luis1\Downloads"
nombre = "2-OTRA-4-STEE"

datos = sd.ScanDataFile.open(path + "\scan_" + nombre + "_scan.NPY") #20x20



for i in range(0,len(datos)):
    ida_img = datos[0][0]
    graficar(ida_img,2/40,0,ida_img.max())
#ida_img = datos[0][0]
#graficar(ida_img,10/200, 0, ida_img.max())
#plt.savefig(full_path)
#%%
selector = ROITraceSelector(datos, channel_in_elem=0, display_frame=0)

#%%
selector = SBRSelector(ida_img)

#%%

img = datos[0][0]   # o la imagen que quieras
x, p = line_profile_interactive(img, pixel_size_um=2/40, linewidth=1)






