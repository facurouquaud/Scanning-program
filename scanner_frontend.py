#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:22:59 2024

This module provides a graphical user interface (GUI) application for scanning
and imaging using PyQt5 and Pyqtgraph. In summary, it allows the user to control
the scan parameters, receive and visualize data, and save the data in different
formats.

This was created to be a Frontend part that can work with multiple scanner Backends,
meaning that it does not handle communication with the hardware devices. The
scanner_class and _scanner have to be defined for each scanner, such as Adwin or
NIDAQ.

The main class is Frontend, while GroupedCheckBoxes and MapRegion handle tasks
that are executed through the main window created through the Frontend

Important Notes
---------------
- The variables for line scan (1D scan) have been added but have not been tested yet
    - Still have to settle a way to switch between NIDAQScanner_2DSignals
      and NIDAQScanner_1DSignals
- Buttonr for XY, XZ, and YZ options were added but are not yet connected properly or
  not implemented in NIDAQ_Scan
"""
import sys
# import os
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton,
    QDockWidget, QComboBox, QRadioButton, QButtonGroup, QLabel,
    QLineEdit, QFormLayout, QSizePolicy, QHBoxLayout, QCheckBox,
    QMessageBox, QInputDialog, QAction, QMenu, QGridLayout, QFileDialog,
)

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QPointF, QRectF
import PyQt5.QtGui as QtGui
import pyqtgraph as pg
import scan_types
import base_scan
import pyqtutils
from map_region import MapWindow
import logging

# Set up logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Local imports
from mocks import mock_scanner
from new_base_scan_2 import NIDAQScan

from typing import Tuple

# Configuration Variables
# --------------------------


# Mapping of scan types to fast and slow axes indices
_scantype_axes_map = {
    scan_types.RegionScanType.XY: [0, 1],
    scan_types.RegionScanType.XZ: [0, 2],
    scan_types.RegionScanType.YZ: [1, 2],
}
_scandata_axes_map = {
    scan_types.RegionScanData.FIRST: [0, 1],
    scan_types.RegionScanData.SECOND: [0, 2],
    scan_types.RegionScanData.BOTH: [1, 2],
}

# Config Globals

# Fixed scan parameters
MAX_SCAN_RANGE_X = 100  # Maximum scan range in X (µm)
MAX_SCAN_RANGE_Y = 100  # Maximum scan range in Y (µm)
MAX_ACCELERATION = 130    # Maximum acceleration (µm/ms²)
MAX_DWELL_TIME = 100     # Maximum dwell time (µs)
MAX_NUM_PIXELS = 500     # Maximum number of pixels in the scan

# Number of lines to keep in plot
_N_PLOT_LINES = 10

# --------------------------
# Classes
# --------------------------


#  Frontend Interface
class FrontEnd(QMainWindow):
    """
    Main window class for scanning application.

    Class creates main GUI window, sets up scanning parameters,
    handles user interactions, and communicates with the scanning backend.
    """

    # Define custom signals
    line_arrived_signal = pyqtSignal(np.ndarray, int)
    scan_end_signal = pyqtSignal()
    scan_start_signal = pyqtSignal()
   # roi_visibility_changed = pyqtSignal(bool)

    _guide_line: pg.InfiniteLine = None
    _scan_params: scan_types.RegionScanParameters
    _center: np.ndarray                           # 3D Center position of scan
    _ROI: pg.ROI = None
    imagen: np.ndarray = None                     # Image scan data array
    _must_restart = False  # Flas we asked the scanner to stop so wwe can restart

    def __init__(self, scanner, *args, **kwargs):
        """ Init FrontEnd window.

        Parameters
        ----------
        scanner_class : class
            Scanning backend class
        """
        super().__init__(*args, **kwargs)
        self.scanner = scanner
        scanner.register_callbacks(
                self.handle_scan_start,
                self._emit_end,
                self._emit_linescan)
        self.single_scan_mode = False

        # Initialize attributes
        self.setWindowTitle("Main Window with Scanner Plot")
        self.setGeometry(100, 100, 1000, 800)
    
     

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(Qt.black))
        self.setPalette(palette)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
            QWidget {
                color: white;
                background-color: #1e1e1e;
            }
            QLabel, QRadioButton, QPushButton {
                color: white;
            }
            QLineEdit {
                color: white;
                background-color: #2c2c2c;
                border: 1px solid #555;
            }
            QDockWidget {
                background-color: #1e1e1e;
                color: white;
            }
        """)
        self.setAutoFillBackground(True)

        self._scanner: base_scan.BaseScan = None

        self.scan_region_window = None
        self.is_scanning = False

        # Data saving attributes
        self.frames = []
        self.last_frame = None

        # Scanning parameters
        self._scan_params = None

        # Center position (x, y, z)
        self._center = np.array((0, 0, 3.,), dtype=np.float64)

        # Create dockable windows and UI components
        self.create_dock_widgets()

        # Update parameters and image extents
        self.update_parameters()
        self.update_image_extents()
        self._update_pixel_size()

        # Connect signals
        self.line_arrived_signal.connect(self.receive_line)
        self.scan_end_signal.connect(self.handle_scan_end)
        self.scan_start_signal.connect(self.handle_scan_start)

        self._map_window = MapWindow((0, 20, 0, 20), .02, parent=self)
        self._map_window.show()

    # ----------------------------
    # Dockable windows
    # ----------------------------
    def create_dock_widgets(self):
        """ Creates dockable widgets for main window.

        Includes: controls and plot area, parameters dock, 1D scan dock,
        positioner dock, and data saving options.
        """
        # Dock widget for controls and plot
        self.controls_plot_dock = QDockWidget("2D Scan", self)
        self.controls_plot_dock = QDockWidget("Scan", self)
        self.controls_plot_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.controls_plot_dock)

        # Widget inside the dock
        controls_plot_widget = QWidget()
        self.controls_plot_dock.setWidget(controls_plot_widget)

        # Layout for controls and plot
        controls_plot_layout = QVBoxLayout()
        controls_plot_widget.setLayout(controls_plot_layout)

        # Create horizontal layout for buttons
        buttons_layout = QHBoxLayout()
        controls_plot_layout.addLayout(buttons_layout)

        # Start/Stop Scan toggle button
        self.start_stop_button = QPushButton("Start Scan", self)
        self.start_stop_button.setCheckable(True)
        self.start_stop_button.clicked.connect(self.start_stop_scan)
        self.start_stop_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        buttons_layout.addWidget(self.start_stop_button)

        # ROI toggle button
        self.roi_button = QPushButton("ROI")
        self.roi_button.setCheckable(True)
        self.roi_button.clicked.connect(self.toggle_roi)
        self.roi_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        buttons_layout.addWidget(self.roi_button)

        # Line Profile toggle button
        self.line_profile_button = QPushButton("Line Profile")
        self.line_profile_button.setCheckable(True)
        self.line_profile_button.clicked.connect(self.toggle_line_profile)
        self.line_profile_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        buttons_layout.addWidget(self.line_profile_button)
        
        # Save image button
        self.save_image_button = QPushButton("Save Image")
        self.save_image_button.setCheckable(True)
        self.save_image_button.clicked.connect(self.toggle_save_image)
        self.save_image_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        buttons_layout.addWidget(self.save_image_button)
        # self.save_button.clicked.connect(self.save_image)
        # grid_layout.addWidget(self.save_button, 8, 2, 1, 2)

        # Create a PyQtGraph GraphicsLayoutWidget
        self.graphics_layout = pg.GraphicsLayoutWidget()
        controls_plot_layout.addWidget(self.graphics_layout)

        # Create a PlotItem with axes
        self.plot_item = self.graphics_layout.addPlot()
        self.plot_item.setAspectLocked(True)
        self.plot_item.setMouseEnabled(x=True, y=True)

        # Set axes labels
        self.plot_item.setLabel('left', 'Y', units='µm')
        self.plot_item.setLabel('bottom', 'X', units='µm')

        # Create ImageItem
        self.image_item = pg.ImageItem()
        self.plot_item.addItem(self.image_item)
        self._guide_line = pg.InfiniteLine(pos=0, angle=0, movable=False,
                                           pen={'color': 'r', 'width': 1},)
        self._guide_line.hide()
        self._ROI = pg.ROI((0, 0,))
        self._ROI.addScaleHandle((0, 0,), (1., 1.,))
        self._ROI.addScaleHandle((1, 1,), (0., 0.,))
        self._ROI.hide()
        # self._ROI.setZValue(10)
        # TODO: remove this, used for developing. Updates ROI *on the fly*
        self._ROI.sigRegionChangeFinished.connect(self.roi_changed)
        self.plot_item.addItem(self._guide_line)
        self.plot_item.addItem(self._ROI)

        # Create histogram for color scaling
        self.histogram = pg.HistogramLUTItem()
        self.histogram.setImageItem(self.image_item)
        self.graphics_layout.addItem(self.histogram)

        # Create the dockable windows
        
        self.create_parameters_dock()
        self.create_positioner_dock()

    def create_parameters_dock(self):
        """
        Creates the dockable window with scan parameter settings.

        Includes: inputs for scan type, ranges, dwell time, acceleration, and
        number of pixels.
        """
        dock = QDockWidget("Set Parameters", self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetFloatable |
                         QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        dock_widget = QWidget()
        dock.setWidget(dock_widget)

        grid_layout = QGridLayout()
        dock_widget.setLayout(grid_layout)

        # 2D Scan type selection
        self.scan_type_widget = QWidget()
        scan_type_layout = QHBoxLayout()
        self.scan_type_widget.setLayout(scan_type_layout)
        # for rb in [self.xy_radio, self.xz_radio, self.yz_radio]:
        #     scan_type_layout.addWidget(rb)
        self.scan_type_group = QButtonGroup()
        self._radiobuttons: dict[QRadioButton, scan_types.RegionScanType] = {}
        for st in scan_types.RegionScanType:
            nrb = QRadioButton(st.value)
            self._radiobuttons[nrb] = st
            self.scan_type_group.addButton(nrb)
            scan_type_layout.addWidget(nrb)
        list(self._radiobuttons.keys())[0].setChecked(True)
        grid_layout.addWidget(QLabel("2D Scan Type:"), 0, 0)
        grid_layout.addWidget(self.scan_type_widget, 0, 1)
        
       # Scan Data selection 
        self.scan_data_widget = QWidget()
        scan_data_layout = QHBoxLayout()
        self.scan_data_widget.setLayout(scan_data_layout)
       
        self.scan_data_group = QButtonGroup()  # Nombre corregido
        self._scan_data_radiobuttons = {}  # Diccionario separado para scan data
       
        for st in scan_types.RegionScanData:
           nrb = QRadioButton(st.value)
           self._scan_data_radiobuttons[nrb] = st
           self.scan_data_group.addButton(nrb)
           scan_data_layout.addWidget(nrb)
       
        # Seleccionar "First" por defecto
        list(self._scan_data_radiobuttons.keys())[0].setChecked(True)
       
        grid_layout.addWidget(QLabel("Scan Data:"), 1, 0)  # Posición ajustada
        grid_layout.addWidget(self.scan_data_widget, 1, 1)
           

        # X Range input
        self.x_range_input = pyqtutils.create_float_lineedit(10)  # Default X range value
        grid_layout.addWidget(QLabel("Fast Range (μm):"), 2, 0)
        grid_layout.addWidget(self.x_range_input, 2, 1)

        # Y Range input
        self.y_range_input = pyqtutils.create_float_lineedit(10)  # Default Y range value
        grid_layout.addWidget(QLabel("Slow Range (μm):"), 3, 0)
        grid_layout.addWidget(self.y_range_input, 3, 1)

        # Dwell time input
        self.dwell_time_input = pyqtutils.create_float_lineedit(0.01)  # Default dwell time
        grid_layout.addWidget(QLabel("Dwell Time (ms):"), 4, 0)
        grid_layout.addWidget(self.dwell_time_input, 4, 1)

        # Acceleration Input
        # self.a_aux_input = pyqtutils.create_float_lineedit("0.1")  # Default a_aux
        # grid_layout.addWidget(QLabel("Acceleration (µm/ms²):"), 5, 0)
        # grid_layout.addWidget(self.a_aux_input, 5, 1)
            
    
        


        # Number of pixels input
        self.num_pixels_input = pyqtutils.create_int_lineedit("500")  # Default number of pixels
        grid_layout.addWidget(QLabel("Number of Pixels:"), 7, 0)
        grid_layout.addWidget(self.num_pixels_input, 7, 1)

        # Pixel size display (read-only)
        pixel_size_label = QLabel("Pixel Size (µm):")
        pixel_size_label.setStyleSheet("QLabel { font-weight: bold; }")  # Optional styling
        self.pixel_size_display = QLabel("")  # will be updated later
        dock_background_color = self.palette().color(self.backgroundRole()).name()
        self.pixel_size_display.setStyleSheet(f"QLabel {{ background-color : {dock_background_color}; }}")

        # Add pixel size label and display in the grid
        grid_layout.addWidget(pixel_size_label, 1, 2)
        grid_layout.addWidget(self.pixel_size_display, 1, 3)

        # Reset parameters button
        self.reset_params_button = QPushButton("Init Params")
        self.reset_params_button.clicked.connect(self.reset_parameters_to_initial)
        grid_layout.addWidget(self.reset_params_button, 2, 2)

        # Apply ROI selection button
        self.apply_roi_button = QPushButton("Apply ROI Selection")
        self.apply_roi_button.clicked.connect(self.request_restart)
        grid_layout.addWidget(self.apply_roi_button, 3, 2)
        
       

        # Connect textChanged signals -> unckeck save scan data checkbox
        self.x_range_input.textChanged.connect(self.changed_px_size)
        self.y_range_input.textChanged.connect(self.changed_px_size)
        self.num_pixels_input.textChanged.connect(self.changed_px_size)

        # Store initial parameters for resetting
        self._initial_params = {
            'scan_type': scan_types.RegionScanType.XY.value,
            'scan_data': scan_types.RegionScanData.FIRST.value,
            'x_range': self.x_range_input.text(),
            'y_range': self.y_range_input.text(),
            'dwell_time': self.dwell_time_input.text(),
            # 'acceleration': self.a_aux_input.text(),
            'num_pixels': self.num_pixels_input.text()}

    def create_positioner_dock(self):
        """ Creates dockable window for positioner controls.

        Includes: controls for single scan, choosing center, full scan option,
        and moving to a specific center.
        """
        dock = QDockWidget("Scan Data and Position Parameters", self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetFloatable |
                         QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        dock_widget = QWidget()
        dock.setWidget(dock_widget)

        layout = QVBoxLayout()
        dock_widget.setLayout(layout)

        # New buttons arranged side by side
        buttons_layout = QHBoxLayout()
        self.single_scan_button = QPushButton("Single Scan")
        # self.scan_line_button = QPushButton("Scan Line")
        self.choose_center_button = QPushButton("Choose Center")
        self.full_scan_checkbox = QCheckBox("Full Scan")

        # Add buttons to layout
        buttons_layout.addWidget(self.single_scan_button)
        # buttons_layout.addWidget(self.scan_line_button)
        buttons_layout.addWidget(self.choose_center_button)
        buttons_layout.addWidget(self.full_scan_checkbox)
        layout.addLayout(buttons_layout)

        # Step size inputs
        self.x_step_input = pyqtutils.create_float_lineedit("0.050")
        self.y_step_input = pyqtutils.create_float_lineedit("0.050")
        self.z_step_input = pyqtutils.create_float_lineedit("0.050")

        # Position
        self._position_inputs: list[QLineEdit] = [
            pyqtutils.create_float_lineedit(f"{d:.3f}") for d in self._center
        ]
        pos_layout = QHBoxLayout()
        for le, name in zip(self._position_inputs, ['X', 'Y', 'Z']):
            pos_layout.addWidget(QLabel(name), stretch=0)
            pos_layout.addWidget(le, stretch=1)
        self.move_to_button = QPushButton("Move to")

        layout.addWidget(QLabel("Central position (µm):"))
        layout.addLayout(pos_layout)
        layout.addWidget(self.move_to_button)

        # Connect buttons to functions
        self.single_scan_button.clicked.connect(self.single_scan)
        # self.scan_line_button.clicked.connect(self.scan_line)
        self.choose_center_button.clicked.connect(self.choose_center)

        #  Update parameters when the move to position button is clicked
        self.move_to_button.clicked.connect(self.move_to_position)
        self.move_to_button.clicked.connect(self.update_parameters)

    def changed_px_size(self):
        """ Uncheck -> Save Scan Data checkbox when pixel size changes."""
        ...

    # ----------------------------
    # Changing/Updating Data and Parameters
    # ----------------------------
    def update_parameters(self):
        """Collects parameters from the UI and updates the scan."""
        checked_button = self.scan_type_group.checkedButton()
        scan_type = self._radiobuttons[checked_button]
        # Obtener modo de datos (First, Second, Both) 
        checked_data_button = self.scan_data_group.checkedButton()
        scan_data = self._scan_data_radiobuttons[checked_data_button]
        # Get X and Y ranges (fast/slow)
        x_range_value = float(self.x_range_input.text())
        y_range_value = float(self.y_range_input.text())

        # Define start and end points based on scan type
        center = self._center[_scantype_axes_map[scan_type]]
        logger.debug("Center: %s", self._center)

        shift = np.array((x_range_value, y_range_value)) / 2
        start_point = center - shift
        end_point = center + shift

        dwell_time = float(self.dwell_time_input.text())
        num_pixels = int(self.num_pixels_input.text())
        # a_aux = self.a_aux_input.text()

        if (not num_pixels) or (x_range_value < 1E-3) or (x_range_value < 1E-3):
            logger.error("Valores muy pequenos")
            raise ValueError("Range not allowed")

        new_scan_params = scan_types.RegionScanParameters(
            scan_type=scan_type,
            scan_data = scan_data,
            start_point=start_point,
            end_point=end_point,
            center=center,
            dwell_time=dwell_time,
            full_data=self.full_scan_checkbox.isChecked(),
            true_px_num=num_pixels,
        )
        self._scan_params = new_scan_params
        self._update_pixel_size()

    def _update_pixel_size(self) -> float:
        """Calculate the pixel size based on fast range and update display."""
        # TODO: manage pixel size for non-square scan regions
        s_range = self._scan_params.end_point - self._scan_params.start_point
        num_pixels = self._scan_params.true_px_num
        pixel_size = s_range[0] / num_pixels if num_pixels else np.inf
        # print(pixel_size, type(pixel_size), s_range)
        self.pixel_size_display.setText(f"{pixel_size:.4f}")
        return pixel_size

    def reset_parameters_to_initial(self):
        """Resets the parameters to their initial values."""
        # Reset UI inputs to init values
        self.x_range_input.setText(self._initial_params['x_range'])
        self.y_range_input.setText(self._initial_params['y_range'])
        self.dwell_time_input.setText(self._initial_params['dwell_time'])
        # self.a_aux_input.setText(self._initial_params['acceleration'])
        self.num_pixels_input.setText(self._initial_params['num_pixels'])

        # Reset center position
        self._center = np.array((0, 0, 3.), dtype=np.float64)
        for le, coord in zip(self._position_inputs, self._center):
            le.setText(f"{coord:.3f}")

        self.update_parameters()
    def toggle_save_image(self, checked):
        """Handle save image button toggle state."""
        if checked:
            # Mostrar diálogo de guardado
            success = self.save_image_dialog()
            
            # Desactivar el botón independientemente del resultado
            self.save_image_button.setChecked(False)
            
            # Mostrar feedback al usuario
            if success:
                QMessageBox.information(self, "Success", "Image saved successfully!")
            else:
                QMessageBox.warning(self, "Cancelled", "Image was not saved.")

    def save_image_dialog(self):
        """Show save dialog and handle image saving."""
        if self.last_frame is None:
            QMessageBox.warning(self, "No Data", "No scan data available to save.")
            return False
    
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"scan_{timestamp}.png"
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Scan Image",
            default_name,
            "PNG Images (*.png);;TIFF Images (*.tif *.tiff);;All Files (*)"
        )
        
        if not filename:
            return False
        
        try:
            # Normalizar y guardar
            image = Image.fromarray(self.normalize_image(self.last_frame))
            
            # Asegurar extensión correcta
            if not filename.lower().endswith(('.png', '.tif', '.tiff')):
                filename += '.png'
                
            image.save(filename)
            logging.info(f"Image saved to {filename}")
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save image:\n{str(e)}")
            logging.error(f"Image save failed: {e}")
            return False
    def normalize_image(self, data: np.ndarray) -> np.ndarray:
        """Normalize image data to 8-bit format."""
        if data.dtype != np.uint8:
            # Escalar a 0-255
            data_min = np.min(data)
            data_max = np.max(data)
            
            if data_max - data_min > 1e-6:  # Evitar división por cero
                normalized = (data - data_min) / (data_max - data_min) * 255
            else:
                normalized = np.zeros_like(data) * 255
                
            return normalized.astype(np.uint8)
        return data

    def move_to_position(self):
        """Sets the center position according to text boxes."""
        new_center = [float(le.text()) for le in self._position_inputs]
        logger.info("Setting new center position:", new_center)
        self._center[:] = new_center
        # piezo.move_to(new_center)
        self.update_parameters()

    def update_image_extents(self):
        """ Updates extents and pixel size of image before scanning.

        TODO: recenter
        TODO: Check half-pixels
        Corrects per half-pixels
        """
        tr = QtGui.QTransform()
        px_s = self._scan_params.pixel_size
        sp = self._scan_params.start_point
        
        # Set transformation for proper image orientation
        # This is usually sufficient for proper image display
        self.image_item.setTransform(
            tr.scale(px_s, px_s).translate(sp[0]/px_s , sp[1]/px_s))
    
        # ----------------------------
        # Data Handeling/Receiving
        # ----------------------------
    @pyqtSlot()
    def handle_scan_end(self):
        """Manages scan end and updates UI accordingly."""
        logger.info("Scan end")
        self.is_scanning = False
        self.update_scan_button_label(False)
        self._guide_line.hide()

    # @pyqtSlot()
    def handle_scan_start(self, params, shape):  # scan_parameters,
        """Manages scan end and updates UI accordingly."""
        logger.info("Scan start")
        self.is_scanning = True
        self.update_scan_button_label(True)
        self.single_scan_button.setEnabled(False)

        self.imagen = np.zeros(shape)
        self.update_image_extents()
        self.last_frame = None
        self._guide_line.show()

    def start_stop_scan(self):
        """Starts or stops the scan based on the toggle button state."""
        self.update_parameters()
        if self.is_scanning:
            self.scanner.stop_scan()
        else:
            if self.is_scanning:
                logger.warning('Requested to start scanning while scan is active')
                return
            try:
                self.scanner.start_scan(self._scan_params)
            except Exception as e:
                print(e, type(e))

    def _emit_linescan(self, line_data: np.ndarray, lineno: int, last:bool):
        """ Proxy callback to signal line data.

        Parameters
        ----------
        
            line_data : np.ndarray
               Data of current line
            lineno : int
                Line number in scan
        """
        line_data = np.asarray(line_data)
        self.line_arrived_signal.emit(line_data, lineno)
        
        if self.single_scan_mode and lineno >= self._scan_params.true_px_num - 1:
           return True
        return False
    

    def _emit_end(self):
        """E"""
        print(
            "emit enddddddd")
        self.scan_end_signal.emit()
    def process_frame(self):
        # Make a copy of the current frame
        frame = np.copy(self.imagen)
        self.last_frame = frame  # Store the last completed frame
        self._map_window.add_region(frame,
                                  self._scan_params.line_length_fast,
                                  self._scan_params.center,
                                  self._scan_params.dwell_time)

        # FIXME: check size
        frame /= self._scan_params.dwell_time  # Normalize
        frame_range = self._scan_params.line_length_fast

        frame_center = np.array(self._center[
            _scantype_axes_map[self._scan_params.scan_type]
        ])


    @pyqtSlot(np.ndarray, int)
    # def receive_line(self, line_data: np.ndarray, line_number: int):
    #     """
    #     Receives a line of data from the scanning backend and updates image.
    #     The data is already filtered according to the selected scan mode.
    #     """
    #     # Get current scan parameters
    #     scan_data_mode = self._scan_params.scan_data
    #     num_pixels = self._scan_params.true_px_num
    #     print("linea, ", line_number)
    #     self.imagen[:, line_number] = line_data
    #     self.image_item.setImage(self.imagen, autoLevels=False)
    #     self.histogram.imageChanged(autoLevel=True)
    #     self.update_guide_line(line_number)
    
    #     # Check if frame is complete
    #     if line_number == self._scan_params.true_px_num - 1:
    #         frame = np.copy(self.imagen)
    #         self.last_frame = frame
    #         self._map_window.add_region(frame,
    #                                   self._scan_params.line_length_x,
    #                                   self._scan_params.center,
    #                                   self._scan_params.dwell_time)
    
    #         # Normalize
    #         frame /= self._scan_params.dwell_time
    #         frame_center = np.array(self._center[
    #             _scantype_axes_map[self._scan_params.scan_type]
    #         ])
    # ----------------------------
    # ROI
    # ----------------------------
    # TODO: rever l'ogica ROI
    def receive_line(self, line_data: np.ndarray, line_number: int):
        # Get current scan parameters
        scan_data_mode = self._scan_params.scan_data
        num_pixels = self._scan_params.true_px_num
        if scan_data_mode == scan_types.RegionScanData.BOTH:
            # Dual-pass mode: line_data contains both passes interleaved
            if len(line_data) != 2 * num_pixels:
                logger.error(f"Datos en modo BOTH tienen longitud incorrecta: {len(line_data)} (esperaba {2 * num_pixels})")
                return
            # Separar y combinar las dos pasadas
            pass1 = line_data[0:num_pixels]  # Datos de primera pasada
            pass2 = line_data[num_pixels:]  # Datos de segunda pasada
           
            # Combinar promediando ambas pasadas
            combined_line = (pass1 + pass2) / 2.0
       
            # Asignar a la imagen (dimensión lenta, dimensión rápida)
            self.imagen[:,-line_number ] = combined_line
        else:
           # Modo de pasada única (FIRST o SECOND)
           self.imagen[:, -line_number -1] = line_data
       
        # Actualizar visualización
        self.image_item.setImage(self.imagen, autoLevels=False)
        self.histogram.imageChanged(autoLevel=True)
        self.update_guide_line(line_number)
        y_val = self._scan_params.end_point[1] - line_number * self._scan_params.pixel_size
        self._guide_line.setValue(y_val)
        # Procesar frame completo
        if line_number == self._scan_params.true_px_num -1:
            frame = np.copy(self.imagen)
            self.last_frame = frame
            # self.last_frame = self.normalize_image(self.imagen)
            self._map_window.add_region(
            frame,
            self._scan_params.line_length_fast,
            self._scan_params.center,
            self._scan_params.dwell_time )
            # Normalizar y procesar
            frame /= self._scan_params.dwell_time
            frame_center = np.array(self._center[
            _scantype_axes_map[self._scan_params.scan_type]])
    
        
    @pyqtSlot(bool)
    def line_callback(self,data, line_idx, last_line, line_handle):
        current_y = self.scan_params.pixel_size * line_idx
        line_handle.set_ydata([current_y, current_y])  # Mueve la línea
        if last_line:
            line_handle.set_visible(True) 
    def toggle_roi(self, checked: bool):
        """ Toggles visibility of ROI on plot.

        Parameters
        ----------
            checked : bool
        """
        if checked:
            # TODO: remove update=False used for debugging
            # El centro del pixel es lo que nos importa, hacemos shift
            # el ROI inicial es 1/4 más chico que la imagen
            crop = (self._scan_params.line_length_fast / 8.,
                    self._scan_params.line_length_slow / 8)
            shifted_pos = [p - self._scan_params.pixel_size / 2 + s for
                           p, s in zip(self._scan_params.start_point, crop)]
            self._ROI.setPos(shifted_pos, update=False)
            self._ROI.setSize(
                (self._scan_params.line_length_fast ,
                 self._scan_params.line_length_slow ),
                update=True,
            )
            self._ROI.show()
        else:
            self._ROI.hide()

    @pyqtSlot(object)
    def roi_changed(self, roi: object):
        """ (For debugging only)

        Update scan parameters based on ROI

        Parameters
        ----------
            roi : object
        """
        if roi is not self._ROI:
            logger.warning("Unexpected ROI: new ROI detected")
        self.apply_ROI_selection(True)

    @pyqtSlot(bool)
    def apply_ROI_selection(self, checked: bool):
        # lee posición y tamaño en coordenadas del plot (µm)
        roi_pos = self._ROI.pos()   # QPointF en unidades del plot (µm)
        roi_size = self._ROI.size() # QSizeF en µm
        
        f_start = roi_pos.x()
        f_end   = f_start + roi_size.x()
        s_start = roi_pos.y()
        s_end   = s_start + roi_size.y()
        
        axes = _scantype_axes_map[self._scan_params.scan_type]
        
        # Calculate center coordinates correctly
        f_center = (f_start + f_end) / 2.0
        s_center = (s_start + s_end) / 2.0
        
        # Update the center array with correct axis mapping
        self._center[axes[0]] = f_center  # Fast axis center
        self._center[axes[1]] = s_center  # Slow axis center
        
        # Update position inputs with correct values
        self._position_inputs[axes[0]].setText(f"{f_center:.3f}")
        self._position_inputs[axes[1]].setText(f"{s_center:.3f}")
        
        # Update ranges directly with ROI size
        self.x_range_input.setText(f"{roi_size.x():.3f}")
        self.y_range_input.setText(f"{roi_size.y():.3f}")
        
        # IMPORTANT: apply parameters immediately
        self.update_parameters()
        
        # If scanning, restart
        if self.is_scanning:
            self.scanner.stop_scan()
            self.scanner.start_scan(self._scan_params)
            

    def request_restart(self):
        """ Crops region selected by ROi and restarts."""
        if self.is_scanning:
            self.scanner.stop_scan()
            self.update_parameters()
            self.start_stop_scan()

    # ----------------------------
    # Types of Scanning Options
    # ----------------------------
    def single_scan(self):
        """Start a single scan."""
        self.single_scan_mode = True
        self.single_scan_button.setEnabled(False)  # Disable button (avoid chaos)
        # self.start_scan()
        self.start_stop_scan()
        # print("Single Scan button pressed")

    # ----------------------------
    # Other Update/Command Handeling
    # ----------------------------
    def update_guide_line(self, line_number: int):
        """Updates position of 'current position' guide line.

        Parameters
        ----------
        line_number : int
            Current scan line number
        """
        self._guide_line.setValue(line_number * self._scan_params.pixel_size +
                                  self._scan_params.start_point[1])
    

    def update_scan_button_label(self, running: bool):
        """Update start/stop scan button label based on scanning state.

        Parameters
        ----------
            running : bool
        """
        if running:
            self.start_stop_button.setText("Stop Scan")
            self.start_stop_button.setChecked(True)
        else:
            self.start_stop_button.setText("Start Scan")
            self.start_stop_button.setChecked(False)

    def toggle_line_profile(self):
        """ Toggles visibility of line profile on plot."""
        if self.line_profile_line is not None:
            # Remove line profile
            self.plot_item.removeItem(self.line_profile_line)
            self.line_profile_line = None
            self.line_profile_plot.hide()
            self.line_profile_curve = None
        else:
            # Create line profile
            self.line_profile_line = pg.LineSegmentROI([
                self._scan_params.start_point, self._scan_params.end_point], pen='g')
            self.plot_item.addItem(self.line_profile_line)
            self.line_profile_line.sigRegionChanged.connect(self.update_line_profile)

            # Create line profile plot
            self.line_profile_plot = pg.PlotWidget(title="Line Profile")
            self.line_profile_plot.show()
            self.line_profile_curve = self.line_profile_plot.plot()
            self.update_line_profile()

    def update_line_profile(self):
        """
        Updates line profile plot based on current line segment ROI.
        """
        if self.line_profile_line is not None and self.imagen is not None:
            data = self.line_profile_line.getArrayRegion(self.imagen, self.image_item)
            if data is not None:
                x = np.arange(len(data))
                self.line_profile_curve.setData(x, data)

    def choose_center(self, checked):
        """
        Toggles choose center mode based on the button's checked state.

        Parameters
        ----------
            checked : bool
        """
        # FIXME: Esto no anda por click != check. Parece ser para seleccionar el centro
        # cuando se usa el click
        print(f"{checked=}")
        if checked:
            self.choose_center_mode = True
            # Connect the scene's mouse click signal
            self.plot_item.scene().sigMouseClicked.connect(self.plot_click)
            logger.info("Choose Center mode activated")
        else:
            self.choose_center_mode = False
            try:
                self.plot_item.scene().sigMouseClicked.disconnect(self.plot_click)
            except TypeError:
                pass
            logger.info("Choose Center mode deactivated")

    def plot_click(self, event):
        """
        Handles mouse click events on plot when choosing center.

        Parameters
        ----------
        event : Event
            Mouse click
        """
        if not self.choose_center_mode:
            return
        # Get mouse click position in scene coordinates
        pos = event.scenePos()
        mouse_point = self.plot_item.vb.mapSceneToView(pos)
        x = mouse_point.x()
        y = mouse_point.y()

        # Update central position
        axes = _scantype_axes_map[self._scan_params.scan_type]
        self._center[axes[0]] = x
        self._center[axes[1]] = y
        # Update the position inputs
        self._position_inputs[axes[0]].setText(f"{x:.3f}")
        self._position_inputs[axes[1]].setText(f"{y:.3f}")

        self.update_parameters()

        self.choose_center_button.setChecked(False)
        self.choose_center_mode = False
        try:
            self.plot_item.scene().sigMouseClicked.disconnect(self.plot_click)
        except TypeError:
            pass
        logger.info("New center selected at (%.3f, %.3f)", x, y)

    # ----------------------------
    def closeEvent(self, event):
        """
        Handles window close event to ensure threads are stopped
        and resources are cleaned up.

        Parameters
        ----------
        event : QCloseEvent
            Close event.
        """
        if self.scanner:
            self.scanner.stop_scan()
        # if self.variable_frames_file is not None:
        #     self.variable_frames_file.close()
        #     self.variable_frames_file = None
        self._map_window.close()
        super().closeEvent(event)

if __name__ == "__main__":
    # scanner = mock_scanner()
    scanner = NIDAQScan()
    app = QApplication(sys.argv)
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    # window = FrontEnd(NIDAQScanner_Manager.NIDAQScanner_2DSignals)
    window = FrontEnd(scanner)

    window.show()
    window.raise_()
    window.activateWindow()

    sys.exit(app.exec_())
    # app.exec_()
    app.quit()