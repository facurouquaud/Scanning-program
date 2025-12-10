#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:28:11 2025

@author: azelcer
"""
from __future__ import annotations
from PyQt5.QtWidgets import (QMainWindow as _QMainWindow, QWidget as _QWidget,
                             QVBoxLayout as _QVBoxLayout, QHBoxLayout as _QHBoxLayout,
                             QPushButton as _QPushButton, )
from PyQt5 import QtGui as _QtGui
import numpy as _np
import logging as _lgn
import pyqtgraph as _pg
from memoimage import MemoImage as _MemoImage
from bounded_roi import BoundedROI as _ROI


_lgr = _lgn.getLogger(__name__)


class MapWindow(_QMainWindow):
    """
    Creates a map window that allows the user to map out the entire scanning region.

    Everytime scan region button is pressed, the current zoomed in scan data
    will be added to the this window. The scan data will be normalized and
    resized before being added to the map window.
    """

    def __init__(
        self,
        image_extents_um: tuple[float, float, float, float],
        image_pixel_size_um: float,
        *args, **kwargs,
    ):
        """
        Initializes the MapRegion window.

        Parameters
        ----------
            image_extents_um: tuple[float, float, float, float]
                extents of the scan area: x_min, x_max, y_min, y_max (ver si cambiar)
            image_pixel_size_um: float
        """
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Full Region")
        self._pixel_size_um = image_pixel_size_um
        x_size_um = image_extents_um[1] - image_extents_um[0]
        y_size_um = image_extents_um[3] - image_extents_um[2]
        self._memoimage = _MemoImage(x_size_um, y_size_um, image_pixel_size_um)
        self.setup_GUI()

    def setup_GUI(self):
        """Initializes interface layout and image plot components."""
        # Create a central widget
        central_widget = _QWidget()
        self.setCentralWidget(central_widget)

        # Create layout
        layout = _QVBoxLayout()
        central_widget.setLayout(layout)

        # Graphics
        self.graphics_layout = _pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics_layout)
        self.plot_item = self.graphics_layout.addPlot(row=0, col=0)
        self.plot_item.setAspectLocked(True)
        self.plot_item.setMouseEnabled(x=True, y=True)
        self.plot_item.setLabel("left", "Y", units="µm")
        self.plot_item.setLabel("bottom", "X", units="µm")
        self.image_item = _pg.ImageItem()
        self.plot_item.addItem(self.image_item)
        self.histogram = _pg.HistogramLUTItem()
        self.histogram.setImageItem(self.image_item)
        self.graphics_layout.addItem(self.histogram, row=0, col=1)
        self.update_image_transform()

        # Buttons
        btn_layout = _QHBoxLayout()
        reset_btn = _QPushButton("Clear")
        reset_btn.clicked.connect(self.reset_contents)
        btn_layout.addWidget(reset_btn)
        layout.addLayout(btn_layout)
        # Create ROI outline
        self._create_roi()

    def _create_roi(self):
        self._ROI = _ROI((0, 0,))
        # self._ROI.addScaleHandle((0, 0,), (1., 1.,))
        # self._ROI.addScaleHandle((1, 1,), (0., 0.,))
        self._ROI.hide()
        # self._ROI.setZValue(10)
        self.plot_item.addItem(self._ROI)
        # self._ROI.sigRegionChanged.connect(self._update_parent_roi)

    def update_roi(self, center: tuple, data_range: float):
        """Updates ROI position."""
        self._ROI.setSize((data_range, data_range), update=False, finish=False)
        self._ROI.setPos(center[0] - data_range/2, center[1] - data_range/2, update=False)
        self._ROI.show()

    def reset_contents(self):
        self._memoimage.clear()
        self._update_image()

    def update_image_transform(self):
        """
        Aligns display of full_region_data with the physical dimensions.
        Scales the image so that each pixel corresponds to the
        correct physical size.
        """
        # Create a QTransform to scale the image item
        transform = _QtGui.QTransform()
        transform.scale(self._pixel_size_um, self._pixel_size_um)
        self.image_item.setTransform(transform)

    def add_region(
        self,
        scan_data: _np.ndarray,
        data_scan_range: float,
        scan_data_center: tuple,
        dwell_time_us,
    ):
        """
        Resizes and places scan_data into equivalent region in full_region_data.
        The region's size and location are determined from scan_data_center
        and data_scan_range in micrometers.

        Parameters
        ----------
        scan_data : numpy.ndarray
            The 2D array representing the scan data to be inserted.
        data_scan_range : float
            The physical size (range) of the scan data in micrometers (µm).
        scan_data_center : tuple of float
            The physical center position (x, y) of the scan data in micrometers (µm).

        NOTE: this was created assuming we are not working with a square region
        """

        # Validate scan_data (just in case)
        if scan_data is None or scan_data.size == 0:
            _lgr.warning("There is no scan data provided.")
            return
        px_size_um = data_scan_range / scan_data.shape[0]
        self._memoimage.add_region(scan_data, px_size_um, *scan_data_center, dwell_time_us)
        self.update_roi(scan_data_center, data_scan_range)
        self._update_image()

    def _update_image(self):
        """Update the image item to display the new data."""
        # autoLevels=False to prevent automatic adjustment of image intensity
        self.image_item.setImage(self._memoimage.data, autoLevels=False)
        self.histogram.imageChanged(autoLevel=True)

    def closeEvent(self, event):
        """
        Parameters
        ----------
            event : QCloseEvent
        """
        super().closeEvent(event)
