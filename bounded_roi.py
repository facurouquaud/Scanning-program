#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 20:06:19 2025

@author: azelcer
"""
from PyQt5.QtWidgets import QWidget as _QWidget
from PyQt5.QtCore import pyqtSlot as _pyqtSlot
from pyqtgraph import ROI as _ROI
from threading import Lock as _Lock
from weakref import WeakSet as _WeakSet


class _BoundedROIManager(_QWidget):
    _ROIs: _WeakSet = _WeakSet()
    _in_event: _Lock = _Lock()  # other "sensible" solution would be disconnect/reconnect

    def add_ROI(self, new_roi: _ROI):
        self._ROIs.add(new_roi)
        new_roi.sigRegionChanged.connect(self._ROI_changed)
        new_roi.sigRegionChangeFinished.connect(self._ROI_change_finished)

    @_pyqtSlot(object)
    def _ROI_changed(self, origin_ROI: _ROI):
        if self._in_event.acquire(blocking=False):
            roi_pos = origin_ROI.pos()
            roi_size = origin_ROI.size()
            for ROI in self._ROIs:
                if ROI is origin_ROI:
                    continue
                ROI.setPos(roi_pos, update=False)
                ROI.setSize(roi_size, update=True, finish=False)
            self._in_event.release()

    @_pyqtSlot(object)
    def _ROI_change_finished(self, origin_ROI: _ROI):
        if self._in_event.acquire(blocking=False):
            roi_pos = origin_ROI.pos()
            roi_size = origin_ROI.size()
            for ROI in self._ROIs:
                if ROI is origin_ROI:
                    continue
                ROI.setPos(roi_pos, update=False)
                ROI.setSize(roi_size)
            self._in_event.release()


class _BoundedROIType(_ROI.__class__):
    _manager: _BoundedROIManager = None

    def __call__(cls, *args, **kwargs):
        if not _BoundedROIType._manager:  # Not thread safe
            _BoundedROIType._manager = _BoundedROIManager()
        new_roi = super(_BoundedROIType, cls).__call__(*args, **kwargs)
        _BoundedROIType._manager.add_ROI(new_roi)
        return new_roi


class BoundedROI(_ROI, metaclass=_BoundedROIType):
    """ROIs that emit signals to stay synchronized."""
    ...
