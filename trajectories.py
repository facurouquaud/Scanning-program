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
