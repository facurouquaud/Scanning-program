#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:16:02 2024

@author: azelcer
"""
from __future__ import annotations
from enum import Enum as _Enum
from base_scan import BaseScan, ScanModeInfo
import scan_parameters
import numpy as _np
import threading
import dataclasses
import time
import logging as _lgn


_lgr = _lgn.getLogger(__name__)


class ScanModes(ScanModeInfo, _Enum):
    IDA = "Ida solo", 1, ["Imagen Ida",]
    IDAVUELTA = "Ida y vuelta", 2, ["Imagen Ida", "Imagen vuelta"]

_SCAN_MODES ={
    mode.description: mode
    for mode in ScanModes
    }

class _mockThread(threading.Thread):
    def __init__(self,
                 line_callbacks,
                 end_callbacks,
                 pars,
                 scan_mode: ScanModes,
                 *args, **kwargs):
        """Prepare thread for running."""
        super().__init__(*args, **kwargs)
        self._pars = pars
        self._cb = line_callbacks
        self._end_cb = end_callbacks
        self._must_stop = False
        self._scan_type = pars.scan_type
        self._line_delay = pars.dwell_time * (pars.true_px_num * 2 + 0) * 1E-6
        self._fast_pos = _np.linspace(pars.start_point[0], pars.end_point[0], pars.true_px_num)
        self._slow_pos = _np.linspace(pars.start_point[1], pars.end_point[1], pars.true_px_num)
        self._tiempos = _np.full((pars.true_px_num,), pars.dwell_time)
        self._scan_mode = scan_mode

    def run(self):
        n_linea = 0
        max_lineno = self._pars.true_px_num - 1
        slow_step = self._pars.pixel_size
        while not self._must_stop:
            line_data = self.scan_line(n_linea)
            try:
                for func in self._cb:
                    if func(line_data, n_linea, n_linea == max_lineno):
                        self._must_stop = True
            except Exception as e:
                _lgr.error("Excepcion (%s) en callback: %s", type(e), e)
                # TODO: mejorar handling.
                self._must_stop = True
            n_linea += 1
            n_linea %= (max_lineno + 1)
        for cb in self._end_cb:
            try:
                cb()  # ????
            except Exception as e:
                print("Error en callback de detención:", e, type(e))
        _lgr.debug("""Salimos del thread""")

    def scan_line(self, n_linea: int):
        time.sleep(self._line_delay)
        fp = self._fast_pos - 10
        sp = self._slow_pos[n_linea] - 10
        if self._scan_type == scan_parameters.RegionScanType.XZ:
            data = (_np.sinc(_np.hypot(fp, sp))**2) * 4 * self._tiempos
        elif self._scan_type is scan_parameters.RegionScanType.YZ:
            data = _np.hypot(fp, sp)**2 * self._tiempos
        else:
            data = _np.sin(fp)**2 * self._tiempos
        if self._scan_mode is ScanModes.IDAVUELTA:
            data = _np.stack((data, data,))
        # data += _np.random.rand(data.shape[0]) * _np.sqrt(self._tiempos)
        return data


class mock_scanner(BaseScan):
    _thread = None
    _scan_type = None
    _is_scanning = False
    _stop_requested = False

    def __init__(self, *args):
        self._start_callbacks = []
        self._end_callbacks = []
        self._line_callbacks = []

    def start_scan(self, params: scan_parameters.RegionScanParameters,
                   scan_mode: ScanModes = None):
        if self._is_scanning:
            print("Ya estoy escaneando")
            return False
        if self._thread:
            print("No limpiaste la última vez")
            self.cleanup_scan()
        if scan_mode:
            print("escaneo=", scan_mode)
        self._scan_params = dataclasses.replace(params)
        # chequear límites, etc
        # Llamar callbacks start
        self.sx = params.true_px_num
        self.sy = params.true_px_num
        self._scan_mode = _SCAN_MODES.get(scan_mode, ScanModes.IDA)
        print("scan mode is", scan_mode, self._scan_mode)
        n_layers = 1 if self._scan_mode is ScanModes.IDA else 2
        for cb in self._start_callbacks:
            try:
                cb(params, (self.sx, self.sy), n_layers)
            except Exception as e:
                print("Error inicializando escaneo:", e)
                # FIXME: deberíamos hacer un unroll de los starts con ends
                raise
        self._init_thread()
        self._is_scanning = True
        return True

    def get_extents(self) -> tuple[tuple[float, float]]:
        return ((0, 20),) * 3

    def get_detectors(self) -> list[str]:
        return [f"Detector {n}" for n in range(4)]

    def get_scan_modes(self) -> list[ScanModeInfo]:
        return [modo for modo in ScanModes]
        # return [modo.value for modo in ScanModes]

    def register_callbacks(
            self,
            scan_start_callback: callable,
            scan_end_callback: callable,
            line_callback: callable):
        if scan_start_callback:
            self._start_callbacks.append(scan_start_callback)
        if scan_end_callback:
            self._end_callbacks.append(scan_end_callback)
        if line_callback:
            self._line_callbacks.append(line_callback)

    def _init_thread(self):
        self._thread = _mockThread(
            self._line_callbacks,
            self._end_callbacks,
            self._scan_params,
            self._scan_mode,
            )
        self._thread.start()

    def stop_scan(self):
        if not self._thread:
            print("No hay hilo activo.")
            return False
        self._thread._must_stop = True
        # self._thread.join()
        self._is_scanning = False
        # self._thread = None

    def cleanup_scan(self):
        if self._thread:
            if self._thread.is_alive():
                print("Ouch")
            self._thread.join()
            self._thread = None
