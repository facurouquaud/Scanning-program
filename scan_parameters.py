#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date: Thu Oct 3 11:23:06 2024

Module that defines data structures for scanning parameters and callbacks.

This module provides classes for exchanging information between the
frontend and backends of scanning systems.
"""
from __future__ import annotations
from dataclasses import dataclass as _dataclass, asdict as _asdict
from enum import Enum as _Enum
import json as _json
import numpy as _np
# from typing import Callable as _Callable


# 2D scans

# ScanCallback = _Callable[[_np.ndarray, int], bool]
class RegionScanType(str, _Enum):
    """ Enumeration of region scan types. """
    XY = 'XY'
    XZ = 'XZ'
    YZ = 'YZ'

    def __repr__(self):
        return f'"{self.value}"'


class _NumpyEncoder(_json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        return super().default(obj)


@_dataclass
class RegionScanParameters:
    """Parameters for region (2D) scan.

    All distances are in µm, all times are in µs

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
    line_length_fast : float
        Length of scan in the fast direction (µm)
    line_length_slow : float
        Length of scan in the slow direction (µm).
    sample_rate : float
        Sample rate based on dwell time (Hz)
    """
    scan_type: RegionScanType
    start_point: tuple[float, float]
    end_point: tuple[float, float]
    center: tuple[float, float]
    dwell_time: float
    true_px_num: int
    a_aux: int
    acceleration: float

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
        return (self.dwell_time * 1E-6)**-1

    def save_to(self, filename: str):
        with open(filename, "wt")as fd:
            _json.dump(_asdict(self), fd, cls=_NumpyEncoder)


if __name__ == "__main__":
    a = RegionScanParameters(RegionScanType.XY, (9, 8), (88,99), (0,0), 100, 14, 14, 1.111)
    print(_asdict(a))
