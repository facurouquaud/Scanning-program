#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 14:47:13 2025

@author: azelcer
@author: facur
"""
import numpy as np
from typing import Tuple


def generate_trajectory(
    current_position: Tuple,
    fast_f: float,
    slow_f: float,
    dwell_time: float,
    a_max_fast: float,
    a_max_slow: float,
) -> Tuple:
    """Generate trajectory with independent axis timing.

    If one axis arrives before the other, it waits on the same place:
    accelerations are obeyed.

    Parameters
    ----------
        current_position: Tuple[float, float]
            fast, slow initial position in µm
        fast_f, slow_f: float
            end points, in µm
        dwell_time: float
            tiem per pixel in µs
        a_max_fast, a_max_slow: float > 0
            accelerations, in µm/µs²

    Returns
    -------
        time per pixel:  numpy.ndarray
        fast axis positions:  numpy.ndarray
        slow axis positions:  numpy.ndarray
        n_points:  int
    """
    assert a_max_fast > 0, "Aceleracion inválida"
    assert a_max_slow > 0, "Aceleracion inválida"

    fast_0, slow_0 = current_position
    dfast = abs(fast_f - fast_0)
    dslow = abs(slow_f - slow_0)

    # Tiempo de movimiento para cada eje
    t_fast = 2 * np.sqrt(dfast / a_max_fast) if dfast > 0 else 0.0
    t_slow = 2 * np.sqrt(dslow / a_max_slow) if dslow > 0 else 0.0

    # Signos de movimiento
    s_fast = np.sign(fast_f - fast_0) if dfast > 0 else 0
    s_slow = np.sign(slow_f - slow_0) if dslow > 0 else 0

    # Tiempo total = el más largo de ambos
    t_total = max(t_fast, t_slow)
    t_end = t_total
    t = np.arange(0, t_end + dwell_time, dwell_time)  # FIXME: linspace?

    fast_back = np.full_like(t, fast_f)
    slow_back = np.full_like(t, slow_f)

    # --- EJE X ---
    if s_fast != 0:
        t_half_fast = t_fast / 2
        mask_move_fast = t < t_fast
        mask_acc_fast = t < t_half_fast
        mask_dec_fast = mask_move_fast & ~mask_acc_fast

        # Aceleración
        fast_back[mask_acc_fast] = (
            fast_0 + 0.5 * s_fast * a_max_fast * t[mask_acc_fast]**2
            )

        # Desaceleración
        t2 = t[mask_dec_fast] - t_half_fast
        v_fast = a_max_fast * t_half_fast
        fast_back[mask_dec_fast] = (
            fast_0
            + 0.5 * s_fast * a_max_fast * t_half_fast**2
            + s_fast * v_fast * t2
            - 0.5 * s_fast * a_max_fast * t2**2
        )

    # --- EJE Y ---
    if s_slow != 0:
        t_half_slow = t_slow / 2
        mask_move_slow = t < t_slow
        mask_acc_slow = t < t_half_slow
        mask_dec_slow = mask_move_slow & ~mask_acc_slow

        t1 = t[mask_acc_slow]
        slow_back[mask_acc_slow] = slow_0 + 0.5 * s_slow * a_max_slow * t1**2

        t2 = t[mask_dec_slow] - t_half_slow
        v_slow = a_max_slow * t_half_slow
        slow_back[mask_dec_slow] = (
            slow_0
            + 0.5 * s_slow * a_max_slow * t_half_slow**2
            + s_slow * v_slow * t2
            - 0.5 * s_slow * a_max_slow * t2**2
        )

    return t, fast_back, slow_back, len(t)


def finish_scan(
        t_0: float, fast_0: float, slow_0: float,
        fast_f: float, slow_f: float, dwell_time: float,
        a_max_fast: float, a_max_slow: float) -> Tuple:
    """Generate flyback trajectory.

    Se mueve sólo en el eje lento, siguiendo la aceleración dada.
    """
    a_slow = a_max_slow
    dslow = abs(slow_f - slow_0)
    # t_fast = 2 * np.sqrt(dfast / a_max_fast)
    t_slow = 2 * np.sqrt(dslow / a_slow)

    # a_fast = a_max_fast
   
    # t_total = max(t_fast, t_slow)
    t_total = t_slow

    t_end = t_total + t_0
    t = np.arange(t_0, t_end + dwell_time, dwell_time)
    n_points = len(t)
    t_half = t_total / 2
    # fast_back = np.zeros(n_points)
    slow_back = np.zeros(n_points)
    # s_fast = 1 if fast_f > fast_0 else -1
    s_slow = 1 if slow_f > slow_0 else -1
    t_rel = t - t_0

    # First half of movement
    mask1 = t_rel < t_half
    t1 = t_rel[mask1]
    # fast_back[mask1] = fast_0 + 0.5 * s_fast * a_fast * t1**2
    slow_back[mask1] = slow_0 + 0.5 * s_slow * a_slow * t1**2

    # Second half of movement
    mask2 = t_rel >= t_half
    t2 = t_rel[mask2] - t_half
    v_slow = a_slow * t_half

    slow_back[mask2] = (slow_0 + 0.5 * s_slow * a_slow * t_half**2 +
                        s_slow * v_slow * t2 - 0.5 * s_slow * a_slow * t2**2)
    fast_back = np.full_like(t, fast_0)
    return t, fast_back, slow_back, n_points

def scanning_2D(n_lines: int, fast_0: float, slow_0: float,
                dwell_time: float, n_pix_acc: int, n_pix: int,
                acc: float, v_f: float, px_size):
    """Generate square scanning trajectory with independent axis timing.

    The residence time in each pixel, including acceleration pixels, is fixed.

    Parameters
    ----------
        n_lines: int
            number of scan lines (slow-axes values)
        fast_0, slow_0: float
            initial positions in µm
        dwell_time: float
            residence time in each pixel
        n_pix_acc: int
            number of pixels used to attain the desired scanning speed
        n_pix: int
            number of pixel per scan lines (fast-axes values)
        acc: float
        v_f: float
        px_size
    """
    if n_pix_acc <= 0 or n_pix <= 0:
        raise ValueError("n_pix_acc and n_pix must be positive")

    n_points = 4 * n_pix_acc + 2 * n_pix
    t_local = np.arange(n_points) * dwell_time
    idx_pix = np.arange(n_points)
    t_line_duration = n_points * dwell_time

    fast = np.zeros_like(t_local)
    slow = np.full_like(t_local, slow_0)  # Constant slow for entire line

    # Aceleración inicial
    mask1 = idx_pix < n_pix_acc
    fast[mask1] = -fast_0 + 0.5 * acc * (idx_pix[mask1] * dwell_time) ** 2

    # Velocidad constante
    mask2 = (idx_pix >= n_pix_acc) & (idx_pix <= n_pix + n_pix_acc)
    fast[mask2] = (
        -fast_0 + 0.5 * acc * ((n_pix_acc * dwell_time) ** 2) +
        v_f * (idx_pix[mask2] - n_pix_acc) * dwell_time
        )

    # Deceleración positiva
    mask3 = (idx_pix > n_pix + n_pix_acc) & (idx_pix <= n_pix + 2 * n_pix_acc)
    t_dec = (idx_pix[mask3] - (n_pix + n_pix_acc)) * dwell_time
    fast[mask3] = (
        -fast_0 + 0.5 * acc * ((n_pix_acc * dwell_time) ** 2) +
        v_f * n_pix * dwell_time + v_f * t_dec - 0.5 * acc * t_dec ** 2
        )

    # Aceleración negativa
    mask4 = (idx_pix >= n_pix + 2 * n_pix_acc) & (idx_pix < n_pix + 3 * n_pix_acc)
    t_acc_neg = (idx_pix[mask4] - (n_pix + 2 * n_pix_acc)) * dwell_time
    fast[mask4] = fast_0 + v_f * (n_pix * dwell_time) - 0.5 * acc * (t_acc_neg) ** 2

    # Velocidad negativa
    mask5 = (idx_pix >= n_pix + 3 * n_pix_acc) & (idx_pix <= 2 * n_pix + 3 * n_pix_acc)
    fast[mask5] = (
        fast_0 + v_f * (n_pix * dwell_time) -
        0.5 * acc * ((n_pix_acc * dwell_time) ** 2) -
        v_f * (idx_pix[mask5] - (n_pix + 3 * n_pix_acc)) * dwell_time
        )

    mask6 = (2 * n_pix + 4 * n_pix_acc >= idx_pix) & (idx_pix >= 2 * n_pix + 3 * n_pix_acc)
    t_dec_final = (idx_pix[mask6] - (2 * n_pix + 3 * n_pix_acc)) * dwell_time
    
    # El eje rápido sigue su deceleración normal
    fast[mask6] = (fast_0 + v_f * (n_pix * dwell_time) -
                   0.5 * acc * ((n_pix_acc * dwell_time) ** 2) -
                   v_f * n_pix * dwell_time - v_f * t_dec_final +
                   0.5 * acc * t_dec_final ** 2)
    
    # IMPORTANTE: En la deceleración final, el eje lento comienza a moverse
    # Calculamos la fracción de tiempo de la deceleración que ha transcurrido
    t_dec_total = n_pix_acc * dwell_time  # Tiempo total de deceleración
    if t_dec_total > 0:
        fraction = t_dec_final / t_dec_total
        # El movimiento en Y ocurre linealmente durante la deceleración
        # Comienza en slow_0 y termina en slow_0 - px_size
        slow[mask6] = slow_0 - px_size * fraction

    # N líneas de escaneo
    num_points = len(t_local)
    t_offsets = np.arange(n_lines) * num_points * dwell_time
    t_total = np.tile(t_local, n_lines) + np.repeat(t_offsets, num_points)
    

    fast_total = np.tile(fast, n_lines)
    

    slow_step = px_size
    slow_shifts = slow_0 - np.arange(n_lines) * slow_step
    slow_total = np.zeros_like(t_total)
    
    for i in range(n_lines):
        start_idx = i * num_points
        end_idx = (i + 1) * num_points
        
        # Para líneas que no son la última
        if i < n_lines - 1:
            # El valor en Y al inicio de la línea (antes de mask6)
            y_start = slow_shifts[i]
            # El valor en Y al final de la línea (después del movimiento en mask6)
            y_end = slow_shifts[i + 1]
            
            # Para los puntos que NO son mask6: valor constante y_start
            # Para los puntos que SÍ son mask6: transición lineal de y_start a y_end
            
            # Primero, copiamos el patrón de slow de una línea
            slow_line = slow.copy()
            
            # Reemplazamos los valores constantes (todos excepto mask6) por y_start
            # Pero mantenemos la transición en mask6, escalada apropiadamente
            # La transición en slow[mask6] va de slow_0 a slow_0 - px_size
            # Queremos que vaya de y_start a y_end
            
            # Calculamos el offset base para esta línea
            base_offset = y_start - slow_0
            
            # Aplicamos el offset base a toda la línea
            slow_line += base_offset
            slow_total[start_idx:end_idx] = slow_line
            
        else:
            # Todos los puntos a slow_shifts[i]
            slow_total[start_idx:end_idx] = slow_shifts[i]

    last_fast = -fast_0
    last_slow = slow_total[-1]
    last_t = t_total[-1]

    # Garantizar que el último punto sea exactamente slow_0
    t_back, fast_back, slow_back, _ = finish_scan(
        last_t, last_fast, last_slow, -fast_0, slow_0, dwell_time, acc/4, acc/4
        )

    # Concatenar la vuelta
    t_total = np.concatenate([t_total, t_back])
    fast_total = np.concatenate([fast_total, fast_back])
    slow_total = np.concatenate([slow_total, slow_back])

    return t_total, fast_total, slow_total, n_points
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def muestra_escaneo(titulo,t,x,y):
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6), sharex=True)
        ax1.scatter(t, x, s=8, color ="black")
        ax1.set_ylabel("Posición X [µm]")
        ax1.set_title(titulo)
        ax1.grid(True)
        ax2.scatter(t, y, s=8, color="black")
        ax2.set_ylabel("Posición Y [µm]")
        ax2.set_xlabel("Tiempo [µs]")
        ax2.grid(True)
        plt.tight_layout()
        plt.show()

    current_position = (15, 15)
    dwell_time = 1
    acc = 1
    # t_back, back_fast_m, back_slow_m, n_back = generate_trajectory(
    #     current_position, 0, 0, dwell_time, a_max_fast=acc*1E-2, a_max_slow=acc
    # )

    # muestra_escaneo(f"Escaneo de {n_back + 1} de escaneo", t_back, back_fast_m, back_slow_m)

    # t_finish, x_finish, y_finish, n_finish = finish_scan(
    #     120,
    #     0, 10, 20, 300, 1, 1, 1)
    # muestra_escaneo(f"finish_scan de {n_finish + 1} de escaneo", t_finish, x_finish, y_finish)
    t_finish, x_finish, y_finish, n_finish = scanning_2D(
        20,  # lineas
        10, 11,  # inicios
        1,  # dwell
        30,  # px_acc
        40,  #px por linea
        1,   # acc
        1,   # vel final
        10)   # pixeles lineales
    muestra_escaneo(f"finish_scan de {n_finish + 1} de escaneo", t_finish, x_finish, y_finish)
