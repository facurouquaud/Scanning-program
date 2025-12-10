#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABC for scanner objects

@author: azelcer
"""
from __future__ import annotations
import scan_parameters
from typing import Callable, Tuple
from abc import ABC as _ABC, abstractmethod as _abstractmethod
from dataclasses import dataclass as _dataclass
import numpy as _np

# Define callback types
# params, shape, n_layers: int
StartCallback = Callable[[object, Tuple[int, int], int], None]
StopCallback = Callable[[], None]
ScanCallback = Callable[[_np.ndarray, int, bool], None]


@_dataclass
class ScanModeInfo:
    """Information about a scan mode.

    This class carries data abput numbers of image obtained from each scan
    from scanners to frontends.

    Args:
        name (str):
            The name of the scan mode.
        images_per_frame (int):
            The number of images acquired per frame.
        images_names (list[str]):
            The name of each image on the frame.

    Example:
        ScanModeInfo("Counter Scan, no idle line", 2, ["Forward", "Backward"])
    """
    description: str
    images_per_frame:  int
    images_names: list[str]


class BaseScan(_ABC):
    """ABC for scanners."""

    @_abstractmethod  # TODO: change parameters name
    def start_scan(self, params: scan_parameters.RegionScanParameters) -> bool:
        """Instruct scanner to start a new scan.

        Return:
            Falsy if an error ocurred, Truish otherwise

        A succesful cal does not mean that a scan has started: this is signaled vía callbacks
        """
        ...

    @_abstractmethod
    def stop_scan(self):
        """Instruct scanner to stop scanning."""
        ...

    @_abstractmethod
    def cleanup_scan(self):
        """Called to clean up a finished scan.

        Should be idempotent.
        """
        ...

    @_abstractmethod
    def register_callbacks(self,
                           scan_start_callback: callable,
                           scan_end_callback: callable,
                           line_callback: callable):
        """Register callbacks for scanning events.

        Callbacks might be called fron different threads.
        """
        ...

    @_abstractmethod
    def get_extents(self) -> tuple[tuple[float, float, float]]:
        """Return ((min_x, min_y, min_z), (max_x, max_y, max_z))."""
        ...

    def get_detectors(self) -> list[str]:
        """TBD."""
        return ["Detector"]

    def get_scan_modes(self) -> list[ScanModeInfo]:
        return [ScanModeInfo("Simple Scan", 1, ["Image",]),]
