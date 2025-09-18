#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 14:47:13 2025

@author: azelcer
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
    """Generate flyback trajectory with independent axis timing.

    Parameters
    ----------
        currenmt
    """
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
    t = np.arange(0, t_end + dwell_time, dwell_time)
    n_points = len(t)

    fast_back = np.empty_like(t)
    slow_back = np.empty_like(t)

    # --- EJE X ---
    if s_fast == 0:
        # Sin movimiento
        fast_back[:] = fast_0
    else:
        t_half_fast = t_fast / 2
        mask_move_fast = (t) < t_fast
        mask1_fast = (t) < t_half_fast
        mask2_fast = mask_move_fast & ~mask1_fast

        # Aceleración
        t1 = t[mask1_fast]
        fast_back[mask1_fast] = fast_0 + 0.5 * s_fast * a_max_fast * t1**2

        # Desaceleración
        t2 = t[mask2_fast] - t_half_fast
        v_fast = a_max_fast * t_half_fast
        fast_back[mask2_fast] = (
            fast_0
            + 0.5 * s_fast * a_max_fast * t_half_fast**2
            + s_fast * v_fast * t2
            - 0.5 * s_fast * a_max_fast * t2**2
        )

        # Mantener en destino tras finalizar
        fast_back[~mask_move_fast] = fast_f

    # --- EJE Y ---
    if s_slow == 0:
        slow_back[:] = slow_0
    else:
        t_half_slow = t_slow / 2
        mask_move_slow = t < t_slow
        mask1_slow = t < t_half_slow
        mask2_slow = mask_move_slow & ~mask1_slow

        t1 = t[mask1_slow]
        slow_back[mask1_slow] = slow_0 + 0.5 * s_slow * a_max_slow * t1**2

        t2 = t[mask2_slow] - t_half_slow
        v_slow = a_max_slow * t_half_slow
        slow_back[mask2_slow] = (
            slow_0
            + 0.5 * s_slow * a_max_slow * t_half_slow**2
            + s_slow * v_slow * t2
            - 0.5 * s_slow * a_max_slow * t2**2
        )

        slow_back[~mask_move_slow] = slow_f

    return t, fast_back, slow_back, n_points

def finish_scan(t_0: float, fast_0: float, slow_0: float,
              fast_f: float, slow_f: float, dwell_time: float,
              a_max_fast: float, a_max_slow: float) -> Tuple:
    """Generate flyback trajectory."""
    dfast = abs(fast_f - fast_0)
    dslow = abs(slow_f- slow_0)
    t_fast = 2 * np.sqrt(dfast / a_max_fast)
    t_slow = 2 * np.sqrt(dslow / a_max_slow)

    a_fast = a_max_fast
    a_slow = a_max_slow
    t_total = max(t_fast, t_slow)

    t_end = t_total + t_0
    t = np.arange(t_0, t_end + dwell_time, dwell_time)
    n_points = len(t)
    t_half = t_total / 2
    fast_back = np.zeros(n_points)
    slow_back = np.zeros(n_points)
    s_fast = 1 if fast_f > fast_0 else -1
    s_slow = 1 if slow_f > slow_0 else -1
    t_rel = t - t_0

    # First half of movement
    mask1 = t_rel < t_half
    t1 = t_rel[mask1]
    fast_back[mask1] = fast_0 + 0.5 * s_fast * a_fast * t1**2
    slow_back[mask1] = slow_0 + 0.5 * s_slow * a_slow * t1**2

    # Second half of movement
    mask2 = t_rel >= t_half
    t2 = t_rel[mask2] - t_half
    v_slow = a_slow * t_half

    fast_back = np.full_like(t, fast_0)

    slow_back[mask2] = (slow_0 + 0.5 * s_slow * a_slow * t_half**2 +
                      s_slow * v_slow * t2 - 0.5 * s_slow * a_slow * t2**2)

    return t, fast_back, slow_back, n_points


def scanning_2D(n_lines: int, fast_0: float, slow_0: float,
                  dwell_time: float, n_pix_acc: int, n_pix: int,
                  acc: float, v_f: float,px_size):
    """Generate scanning trajectory with independent axis timing.

    Parameters
    ----------
        currenmt
    """
    if n_pix_acc <= 0 or n_pix <= 0:
        raise ValueError("n_pix_acc and n_pix must be positive")

    t_line_duration = 4 * (n_pix_acc * dwell_time) + 2 * n_pix * dwell_time
    t_local = np.arange(0, t_line_duration , dwell_time)
    idx_pix = np.arange(len(t_local))
    n_points = 4 * n_pix_acc + 2 * n_pix
    v_slow = v_f / 2

    fast = np.zeros_like(t_local)
    slow = np.full_like(t_local, slow_0)  # Constant slow for entire line

    # Aceleración inicial
    mask1 = idx_pix < n_pix_acc
    fast[mask1] = -fast_0 + 0.5 * acc * (idx_pix[mask1] * dwell_time) ** 2

    # Velocidad constante
    mask2 = (idx_pix >= n_pix_acc) & (idx_pix <= n_pix + n_pix_acc)
    fast[mask2] = -fast_0 + 0.5 * acc * ((n_pix_acc * dwell_time) ** 2) + v_f * (idx_pix[mask2] - n_pix_acc) * dwell_time

    # Deceleración positiva
    mask3 = (idx_pix > n_pix + n_pix_acc) & (idx_pix <= n_pix + 2 * n_pix_acc)
    t_dec = (idx_pix[mask3] - (n_pix + n_pix_acc)) * dwell_time
    fast[mask3] = (-fast_0 + 0.5 * acc * ((n_pix_acc * dwell_time) ** 2) +
                v_f * n_pix * dwell_time + v_f * t_dec - 0.5 * acc * t_dec ** 2)

    # Aceleración negativa
    mask4 = (idx_pix >= n_pix + 2 * n_pix_acc) & (idx_pix < n_pix + 3 * n_pix_acc)
    t_acc_neg = (idx_pix[mask4] - (n_pix + 2 * n_pix_acc)) * dwell_time
    fast[mask4] = fast_0 + v_f * (n_pix * dwell_time) - 0.5 * acc * (t_acc_neg) ** 2

    # Velocidad negativa
    mask5 = (idx_pix >= n_pix + 3 * n_pix_acc) & (idx_pix <= 2 * n_pix + 3 * n_pix_acc)
    fast[mask5] = (fast_0 + v_f * (n_pix * dwell_time) -
                0.5 * acc * ((n_pix_acc * dwell_time) ** 2) -
                v_f * (idx_pix[mask5] - (n_pix + 3 * n_pix_acc)) * dwell_time)

    # Deceleración final
    mask6 = (2 * n_pix + 4 * n_pix_acc >= idx_pix) & (idx_pix >= 2 * n_pix + 3 * n_pix_acc)
    t_dec_final = (idx_pix[mask6] - (2 * n_pix + 3 * n_pix_acc)) * dwell_time
    fast[mask6] = (fast_0 + v_f * (n_pix * dwell_time) -
                0.5 * acc * ((n_pix_acc * dwell_time) ** 2) -
                v_f * n_pix * dwell_time - v_f * t_dec_final +
                0.5 * acc * t_dec_final ** 2)

    # N líneas de escaneo
    num_points = len(t_local)
    t_offsets = np.arange(n_lines) * num_points * dwell_time
    t_total = np.tile(t_local, n_lines) + np.repeat(t_offsets, num_points)

    fast_total = np.tile(fast, n_lines)
    # x_total -= (x_total.max() + x_total.min()) / 2

    slow_step = px_size
    slow_shifts = slow_0 - np.arange(n_lines) * slow_step
    slow_offsets = np.repeat(slow_shifts, num_points)
    # slow_total = np.tile(np.ones_like(slow)*slow_0, n_lines) + slow_offsets
    slow_total = np.tile(slow, n_lines) + (slow_offsets - slow_0)
    # slow_total -= (slow_total.max() + slow_total.min()) / 2

    # le pido que n_lineso haga una subida mas en slow
    # slow_total[-num_points:] = slow_total[-num_points - 1]

    last_fast = fast_total[-1]
    last_slow = slow_total[-1]
    last_t = t_total[-1]

    # Garantizar que el último punto sea exactamente slow_0

    t_back, fast_back, slow_back,_ = finish_scan(last_t, last_fast, last_slow, fast_0, slow_0,dwell_time, acc, acc)

    # Concatenar la vuelta
    t_total = np.concatenate([t_total, t_back])
    fast_total = np.concatenate([fast_total, fast_back])
    slow_total = np.concatenate([slow_total, slow_back])
    # slow_total -= (slow_total.max() + slow_total.min()) / 2
    # slow_total[-num_points:] = slow_total[-num_points]
    # x_total -= (x_total.max() + x_total.min()) / 2

    return t_total, fast_total, slow_total, int(n_points)
