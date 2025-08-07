#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:28:11 2025

@author: azelcer
"""
from PyQt5.QtWidgets import (QMainWindow as _QMainWindow, QWidget as _QWidget,
                             QVBoxLayout as _QVBoxLayout, QHBoxLayout as _QHBoxLayout,
                             QPushButton as _QPushButton, )
from PyQt5 import QtGui as _QtGui
from PyQt5.QtCore import pyqtSlot as _pyqtSlot
import numpy as _np
import logging as _lgn
from PIL import Image as _Image
import pyqtgraph as _pg
from memoimage import MemoImage as _MemoImage


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
        x_size_um =  image_extents_um[1] - image_extents_um[0]
        y_size_um =  image_extents_um[3] - image_extents_um[2]
        self._memoimage = _MemoImage(x_size_um, y_size_um, image_pixel_size_um)
        # Setup window layout and plot
        self.setup_GUI()

        # # Connect main ROI's region change signal to update the ROI outline
        # if self.main_roi is not None:
        #     self.main_roi.sigRegionChanged.connect(self.update_roi)

        # Connect to the parent's roi_visibility_changed signal
        # if self.parent() is not None:
        #     self.parent().roi_visibility_changed.connect(
        #         self.handle_roi_visibility_change
        #     )

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
        reset_btn =  _QPushButton("Clear")
        reset_btn.clicked.connect(lambda x: (self._memoimage.clear(), self._update_image()))
        btn_layout.addWidget(reset_btn)
        self._ROI_btn =  _QPushButton("Show ROI")
        btn_layout.addWidget(self._ROI_btn)
        layout.addLayout(btn_layout)
        # Create ROI outline
        # self.create_roi()

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
        self._update_image()

    def _update_image(self):
        """Update the image item to display the new data."""
        # autoLevels=False to prevent automatic adjustment of image intensity
        self.image_item.setImage(self._memoimage.data, autoLevels=False)
        self.histogram.imageChanged(autoLevel=True)

    # def create_roi(self):
    #     """
    #     Creates ROI outline based on ROI from the FrontEnd window.
    #     """
    #     if self.main_roi is not None:
    #         # Get position and size from main ROI
    #         roi_pos = self.main_roi.pos()
    #         roi_size = self.main_roi.size()
    #         roi_pos = [roi_pos.x(), roi_pos.y()]
    #         roi_size = [roi_size.x(), roi_size.y()]
    #     else:
    #         # Use these values if main_roi is not provided
    #         roi_pos = [self.init_center[0] - 1, self.init_center[1] - 1]
    #         roi_size = [0, 0]

    #     # Create the ROI outline
    #     self.roi_outline = _pg.RectROI(
    #         roi_pos, roi_size, pen=_pg.mkPen("r", width=2), movable=False
    #     )
    #     self.plot_item.addItem(self.roi_outline)

    #     # Set initial visibility based on main ROI's visibility
    #     if self.main_roi is not None and not self.main_roi.isVisible():
    #         self.roi_outline.hide()

    # def update_roi(self):
    #     """
    #     Updates ROI's outline position and size whenever main ROI is moved or
    #     resized.
    #     """
    #     if self.main_roi is not None:
    #         # Retrieve main ROI's position and size
    #         roi_pos = self.main_roi.pos()
    #         roi_size = self.main_roi.size()

    #         if self.main_roi.isVisible():
    #             # Update  ROI outline position
    #             self.roi_outline.setPos(roi_pos.x(), roi_pos.y())
    #             # Update ROI outline size
    #             self.roi_outline.setSize([roi_size.x(), roi_size.y()])
    #             # Ensure ROI outline is visible
    #             if not self.roi_outline.isVisible():
    #                 self.roi_outline.show()
    #         else:
    #             # Hide ROI outline if main ROI is not visible
    #             if self.roi_outline.isVisible():
    #                 self.roi_outline.hide()

    # @_pyqtSlot(bool)
    # def handle_roi_visibility_change(self, is_visible: bool):
    #     """
    #     Handles ROI visibility based on the map_region window

    #      Parameters
    #      ----------
    #      is_visible : bool
    #          Visibility state of main ROI
    #     """
    #     if is_visible:
    #         self.roi_outline.show()
    #     else:
    #         self.roi_outline.hide()

    def closeEvent(self, event):
        """
        Parameters
        ----------
        event : QCloseEvent
        """
        # try:
        #     self.main_roi.sigRegionChanged.disconnect(self.update_roi)
        # except TypeError:
        #     pass
        # # Disconnect parent's ROI visibility signal
        # if self.parent() is not None:
        #     try:
        #         self.parent().roi_visibility_changed.disconnect(
        #             self.handle_roi_visibility_change
        #         )
        #     except TypeError:
        #         pass
        super().closeEvent(event)
