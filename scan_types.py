#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date: Thu Oct 3 11:23:06 2024

Module that defines data structures for scanning parameters and callbacks.

This module provides classes for exchanging information between the
frontend and backends of scanning systems.

Classes
-------
- LineScanType : Enum
    Enumeration of line scan types.
- LineScanParameters : Dataclass
    Parameters for line scan.
- RegionScanType : Enum
    Enumeration of region scan types.
- RegionScanParameters : Dataclass
    Parameters for region (2D) scan.
- ScanCallback : Type alias

"""

from dataclasses import dataclass as _dataclass
from enum import Enum as _Enum
from typing import Callable as _Callable
import numpy as _np


class LineScanType(_Enum):
    """ Enumeration of line scan types."""
    X = 'X'
    Y = 'Y'
    Z = 'Z'


@_dataclass
class LineScanParameters:
    """
    Parameters for a line scan.

    Attributes
    ----------
    scan_type : LineScanType
        Type of line scan (X, Y, or Z).
    start_point : float
        Starting point of scan (µm).
    end_point : float
        Ending point of scan (µm).
    dwell_time : float
        Time per pixel (µs).
    true_px_num : int
        Number of pixels (does not include auxiliary pixels)
    a_aux : float
         Acceleration for auxiliary pixels
    center : float
        Center point of scan (µm).

    Properties
    ----------
    length : float
        Total length of scan (µm).
    line_time : float
        Total time for scan line (µs).
    pixel_size : float
        Size of each pixel (µm).
    """
    scan_type: LineScanType
    end_point: float
    start_point: float
    dwell_time: float
    true_px_num: int
    center: float
    line_number: float
    line_scan: bool = False

    @property
    def length(self) -> float:
        return self.end_point - self.start_point

    @property
    def line_time(self) -> float:
        return self.dwell_time * self.true_px_num

    @property
    def pixel_size(self) -> float:
        return self.length / self.true_px_num


# 2D scans

ScanCallback = _Callable[[_np.ndarray, int], bool]

class RegionScanType(_Enum):
    """ Enumeration of region scan types. """
    XY = 'XY'
    XZ = 'XZ'
    YZ = 'YZ'
class RegionScanData(_Enum):
    FIRST = "First (forward)"
    SECOND = "Second (backward)"
    BOTH = "Both "
    


@_dataclass
class RegionScanParameters:
    """
    Parameters for region (2D) scan.

    Attributes
    ----------
    scan_type : RegionScanType
        Region scan type (XY, XZ, or YZ)
    scan_data : RegionScanData
        Region scan data (first, second, both)
    start_point : Tuple[float, float]
        Starting point of scan (µm)
    end_point : Tuple[float, float]
        Ending point of scan (µm)
    center : Tuple[float, float]
        Center point of scan (µm)
    dwell_time : float
        Time spent per pixel (µs)
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
    full_data: bool = False  # Have to connect in front end 

    @property
    def pixel_size(self) -> float:
        return self.line_length_fast / self.true_px_num

    @property
    def line_length_fast(self) -> float:
        return self.end_point[0] - self.start_point[0]

    @property
    def line_length_slow(self) -> float:
        return self.end_point[1] - self.start_point[1]

    @property
    def sample_rate(self) -> float:
        return (self.dwell_time*1E-6)**-1
