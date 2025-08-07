#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABC for scanner objects

@author: azelcer
"""


import scan_parameters
from abc import ABC as _ABC, abstractmethod as _abstractmethod


class BaseScan(_ABC):
    """ABC for scanners."""

    @_abstractmethod  # TODO: change parameters name
    def start_scan(self, params: scan_parameters.RegionScanParameters):
        ...

    @_abstractmethod
    def stop_scan(self):
        ...

    @_abstractmethod
    def register_callbacks(self,
            scan_start_callback: callable,
            scan_end_callback: callable,
            line_callback: callable):
        ...

    @_abstractmethod
    def get_extents(self) -> tuple[tuple[float, float]]:
         ...

    
