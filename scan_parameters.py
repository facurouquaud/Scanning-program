#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date: Thu Oct 3 11:23:06 2024

Module that defines data structures for scanning parameters and callbacks.

This module provides classes for exchanging information between the
frontend and backends of scanning systems.
"""

from dataclasses import dataclass as _dataclass
from enum import Enum as _Enum
# from typing import Callable as _Callable
import numpy as _np



# 2D scans

# ScanCallback = _Callable[[_np.ndarray, int], bool]

class RegionScanType(_Enum):
    """ Enumeration of region scan types. """
    XY = 'XY'
    XZ = 'XZ'
    YZ = 'YZ'

class RegionScanData(_Enum):
    FIRST = 'FIRST'
    SECOND = 'SECOND'
    BOTH = 'BOTH'
    
@_dataclass
class RegionScanParameters:
    """
    Parameters for region (2D) scan.

    Attributes
    ----------
    scan_type : RegionScanType
        Region scan type (XY, XZ, or YZ)
    start_point : Tuple[float, float]
        Starting point of scan (µm)
    end_point : Tuple[float, float]
        Ending point of scan (µm)
    center : Tuple[float, float]
        Center point of scan (µm)
    dwell_time : float
        Time spent per pixel (ms)
    true_px_num : int
        Number of pixels per line (does not include auxiliary pixels)
    a_aux : float
        Acceleration for auxiliary pixels
    full_data : bool
        Tells when to return full data including acceleration phases
        (Default is False)

    Properties
    ----------
    pixel_size : float
        Size of each pixel (µm)
    line_length_x : float
        Length of scan in x-direction (µm)
    line_length_y : float
        Length of scan in y-direction (µm).
    sample_rate : float
        Sample rate based on dwell time (Hz)
    """
    scan_type: RegionScanType
    scan_data: RegionScanData
    start_point: tuple[float, float]
    end_point: tuple[float, float]
    center: tuple[float, float]
    dwell_time: float
    true_px_num: int
    acceleration: float
    full_data: bool = False  # Have to connect in front end 

    @property
    def pixel_size(self) -> float:
        if self.line_length_fast > self.line_slow:
            return self.line_length_slow / self.true_px_num
        else:
            return self.line_length_fast / self.true_px_num

    @property
    def line_length_fast(self) -> float:
        return self.end_point[0] - self.start_point[0]

    @property
    def line_length_slow(self) -> float:
        return self.end_point[1] - self.start_point[1]
    


    @property
    def sample_rate(self) -> float:
        return (self.dwell_time * 1E-6)**-1
