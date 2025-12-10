#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 11:36:31 2025

@author: azelcer
"""
from __future__ import annotations
import numpy as _np
from scan_parameters import RegionScanParameters
import matplotlib.pyplot as plt

class ScanDataFile:

    _file: None
    # def __init__(self, filename: str, params: RegionScanParameters, overwrite: bool = False):
    #     # Crear el archivo (ojo si ya existe)
    #     # grabar los parametros como coso estucturado (esto después)
    #     ...

    @classmethod
    def open(cls, filename: str) -> ScanDataFile:
        """"Crea un scansataFile desde un archivo"""
        scans = []
        with open(filename, "rb") as f:
            # npy_pars = _np.load(f)
            try:
                while ((data := _np.load(f, allow_pickle=True)) is not None):
                    scans.append(data)
            except (EOFError, IOError):
                ...
        return scans

    @classmethod
    def create(cls, filename: str, params: RegionScanParameters, overwrite: bool = False):
        """"Crea un scansataFile desde un archivo"""
        mode = "xb"
        if overwrite:
            mode = "wb"
        rv = ScanDataFile()
        rv._file = open(filename, mode)
        # pasar params a npoy estucturado y grabar
        # _np.save(f, npypars)

    def close(self):
        if self._file:
            self._file.close()

if __name__ == "__main__":
    pars = None
    # s = ScanDataFile("/tmp/facu", None)
    p = ScanDataFile.open(r"C:\Users\Usuario\Documents\FacuR\231025\calibracion_00_scan.NPY")
    fig = plt.figure()
    plt.imshow(p[0][0])
    ffig = plt.figure()
    plt.imshow(p[0][1])