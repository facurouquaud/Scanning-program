#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:16:02 2024

@author: azelcer
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:16:02 2024

@author: azelcer
"""

from base_scan import BaseScan
import scan_parameters
import numpy as _np
import threading
import dataclasses
import time
import logging as _lgn

_lgr = _lgn.getLogger(__name__)

class _mockThread(threading.Thread):
    def __init__(self, callbacks, pars: scan_parameters.RegionScanParameters, *args, **kwargs):
        """Prepare thread for running."""
        super().__init__(*args, **kwargs)
        self._pars = pars  # Cambiado de self.scan_params a self._pars para consistencia
        self._cb = callbacks
        self._must_stop = False
        self._scan_type = pars.scan_type
        self._scan_data = pars.scan_data
        self._line_delay = pars.dwell_time * (pars.true_px_num * 2 + 0) * 1E-6
        self._fast_pos = _np.linspace(pars.start_point[0], pars.end_point[0], pars.true_px_num)
        self._slow_pos = _np.linspace(pars.start_point[1], pars.end_point[1], pars.true_px_num)
        self._tiempos = _np.full((pars.true_px_num,), pars.dwell_time)

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
                self._must_stop = True
            n_linea += 1
            n_linea %= (max_lineno + 1)
        _lgr.debug("Salimos del thread")

    def scan_line(self, n_linea: int):
        time.sleep(self._line_delay)
        fp = self._fast_pos[n_linea] - 10
        sp = self._slow_pos - 10
        offset = 1
        fp2 = self._fast_pos[n_linea] - 10 - offset
    
        # Primero generamos todos los datos posibles
        line_data_first = None
        line_data_second = None
        
        if self._scan_type == scan_parameters.RegionScanType.XZ:
            line_data_first = (_np.sinc(_np.hypot(fp, sp))**2) * 4 * self._tiempos
            line_data_second = (_np.sinc(_np.hypot(fp2, sp))**2) * 4 * self._tiempos
        elif self._scan_type == scan_parameters.RegionScanType.YZ:
            line_data_first = _np.hypot(fp, sp)**2 * self._tiempos
            line_data_second = _np.hypot(fp2, sp)**2 * self._tiempos
        else:  # XY por defecto
            line_data_first = (_np.sinc(_np.hypot(fp, sp)) + .2172) * self._tiempos
            line_data_second = (_np.sinc(_np.hypot(fp2, sp)) + .2172) * self._tiempos
        
        # Selección del modo idéntica a la NIDAQ
        if self._scan_data.name == 'FIRST':
            return line_data_first
        elif self._scan_data.name == 'SECOND':
            return line_data_second
        elif self._scan_data.name == 'BOTH':
            return _np.concatenate([line_data_first, line_data_second])
        else:
            _lgr.error(f"Modo de escaneo no reconocido: {self._scan_data}")
            return _np.zeros_like(self._fast_pos) * self._tiempos  # Fallback seguro


class mock_scanner(BaseScan):
    _thread = None
    _is_scanning = False

    def __init__(self, *args):
        self._start_callbacks = []
        self._end_callbacks = []
        self._line_callbacks = []
        self._scan_params = None

    def start_scan(self, params: scan_parameters.RegionScanParameters):
        if self._is_scanning:
            print("Ya estoy escaneando")
            return False
            
        self._scan_params = dataclasses.replace(params)
        self.sx = params.true_px_num
        self.sy = params.true_px_num
        
        for cb in self._start_callbacks:
            try:
                cb(params, (self.sx, self.sy))
            except Exception as e:
                print("Error inicializando escaneo:", e)
                
        self._init_thread()
        self._is_scanning = True
        return True

    def get_extents(self) -> tuple[tuple[float, float]]:
        return ((0, 20),) * 3

    def register_callbacks(self, scan_start_callback: callable,
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
            callbacks=self._line_callbacks,
            pars=self._scan_params)
        self._thread.start()

    def stop_scan(self):
        if not self._thread:
            print("No hay hilo activo.")
            return False
            
        self._thread._must_stop = True
        self._thread.join()
        self._thread = None

        for cb in self._end_callbacks:
            try:
                cb()
            except Exception as e:
                print("Error en callback de detención:", e)
                
        self._is_scanning = False