#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 10:25:34 2025

@author: azelcer

ver fit_scan_data del otro archivo.

"""

import numpy as np
from PIL import Image as _Image
import matplotlib.pyplot as plt
import logging as _lgn

_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)


class MemoImage:
    """
    Crea la imagen de referencia (FOV)
    -----
    Parametros:
    x_size_um: float, y_size_um: float tamaño de x e y
    px_size_um: float tamaño de pixel
    """

    def __init__(
        self, x_size_um: float, y_size_um: float, px_size_um: float, *args, **kwargs
    ):
        _lgr.debug(
            "Creando MemoImage de %s por %s µm, con pixeles de %s µm",
            x_size_um,
            y_size_um,
            px_size_um,
        )
        self.px_size_um = px_size_um
        self.x_size_um = x_size_um
        self.y_size_um = y_size_um
        # data tiene la información en cuentas por unidad de tiempo
        self.data = np.zeros(
            (int(x_size_um / px_size_um), (int(y_size_um / px_size_um)))
        )
        print(self.data.shape)

    def add_region(
        self,
        region: np.ndarray,
        px_size_um: float,
        x_center_um: float,
        y_center_um: float,
        dwell_time_us: float,
    ):
        """
        Añade una porción de imagen en las dimensiones de la original,
        localizandola en el lugar deseado.

        Parametros:
        -----------
        region: np.ndarray
            Imagen que tomamos del FOV
        px_size_um: float
            Tamaño de pixel en µm
        x_center_um: float
            posicion x del centro de region, en µm
        y_center_um: float
            posicion y del centro de region, en µm
        dwell_time_us: float
            tiempo por pixel en µs
        """

        # Convertir la región en una imagen de Pillow
        region_image = _Image.fromarray(region)
        # print(f"{px_size_um=} {x_center_um=} {y_center_um=} {dwell_time_us=}")

        # Redimensionar la imagen con el tamaño de píxeles adecuado
        n_pixel_x = max(1, int(region.shape[0] * px_size_um / self.px_size_um))
        n_pixel_y = max(1, int(region.shape[1] * px_size_um / self.px_size_um))

        resized_region = region_image.resize((n_pixel_x, n_pixel_y))

        # Convertir la imagen redimensionada de nuevo a un array numpy
        resized_array = np.array(resized_region) / dwell_time_us

        # Centro en pixeles
        x_index = int(x_center_um / self.px_size_um)
        y_index = int(y_center_um / self.px_size_um)

        # Esquina en pixeles
        x_lim, y_lim = self.data.shape

        # Ajusto la imagen (data)
        x_start_d = max(0, x_index - resized_array.shape[0] // 2)
        x_end_d = min(
            x_index + resized_array.shape[0] // 2 + resized_array.shape[0] % 2, x_lim
        )
        y_start_d = max(0, y_index - resized_array.shape[1] // 2)
        y_end_d = min(
            y_index + resized_array.shape[1] // 2 + resized_array.shape[0] % 2, y_lim
        )

        # Ajusto la imagen (Region)
        x_start_r = max(0, resized_array.shape[0] // 2 - x_index)
        x_end_r = min(
            resized_array.shape[0],
            resized_array.shape[0] - (x_index + resized_array.shape[0] // 2 - x_lim),
        )
        y_start_r = max(0, resized_array.shape[1] // 2 - y_index)
        y_end_r = min(
            resized_array.shape[1],
            resized_array.shape[1] - (y_index + resized_array.shape[1] // 2 - y_lim),
        )

        # Chequeo que la imágen no se vaya de los límites (de lo contr da error)
        if x_end_d <= 0 or y_end_d <= 0 or x_start_d >= x_lim or y_start_d >= y_lim:
            _lgr.debug("Imagen fuera de los límites")
            return
        self.data[x_start_d:x_end_d, y_start_d:y_end_d] = resized_array[
            x_start_r:x_end_r, y_start_r:y_end_r
        ]

    def clear(self):
        """Erase all data."""
        self.data[:] = 0.0


if __name__ == "__main__":
    muestra = np.eye(40, dtype=np.uint8) * 255
    flip = np.fliplr(muestra)
    dwell_time = 200
    img = MemoImage(20, 20, 0.01)

    img.add_region(muestra, 0.132, 4.250, 5.3330, dwell_time)
    img.add_region(flip, 0.1, 20, 0, dwell_time)
    plt.imshow(img.data, cmap="gray")

    memo = MemoImage(20, 20, 0.01)
    region = np.random.randint(
        0, 255, (10, 10), dtype=np.uint8
    )  # imagen 10x10 valores aleatorios
    # region_2 = np.random.randint(0, 255, (10, 10), dtype=np.uint8)

    # Agregar la región al MemoImage
    memo.add_region(
        region, px_size_um=0.1, x_center_um=10, y_center_um=10, dwell_time_us=100
    )
    # memo.add_region(region_2, px_size_um=1, x_center_um=50, y_center_um=50, dwell_time_us=100)
    # Mostrar la imagen resultante
    plt.figure("nueva")
    plt.imshow(memo.data, cmap="gray")
    plt.show()

    plt.imshow(memo.data, cmap="gray")
    # memo.clear()
