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

"""
from __future__ import annotations
import sys
import numpy as np

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QDockWidget,
    QComboBox,
    QRadioButton,
    QButtonGroup,
    QLabel,
    QLineEdit,
    QFormLayout,
    QSizePolicy,
    QHBoxLayout,
    QCheckBox,
    QMessageBox,
    QInputDialog,
    QAction,
    QMenu,
    QGridLayout,
    QFileDialog,
    QAbstractButton,
    QComboBox,
)

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QPointF, QObject
import PyQt5.QtGui as QtGui
import pyqtgraph as pg
import scan_parameters
import base_scan
from utils import pyqtutils
from map_region import MapWindow
import logging
from fname_server import FileNameServer

# Local imports
# from mocks import mock_scanner
from drivers.NIDAQ import NIDAQScan as mock_scanner
# from mocks import mock_scanner as mock_scanner
from bounded_roi import BoundedROI


# Set up logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Configuration Variables
# --------------------------
# Mapping of scan types to fast and slow axes indices
_scantype_axes_map = {
    scan_parameters.RegionScanType.XY: [0, 1],
    scan_parameters.RegionScanType.XZ: [0, 2],
    scan_parameters.RegionScanType.YZ: [1, 2],
}

# Config Globals

# Fixed scan parameters
MAX_SCAN_RANGE_X = 19.5  # Maximum scan range in X (µm)
MAX_SCAN_RANGE_Y = 19.5  # Maximum scan range in Y (µm)
MAX_ACCELERATION = 0.1  # Maximum acceleration (µm/ms²)
MAX_DWELL_TIME = 100  # Maximum dwell time (µs)
MAX_NUM_PIXELS = 200  # Maximum number of pixels in the scan


def _str_incr(txt: str):
    """Incrementa ASCII alfanumericamente SIN checkeo de errores"""
    txt = txt.upper()
    if txt == "":  # braks ordering but allows overflowing without errors
        return "0"
    if txt[-1] == '9':
        return txt[:-1] + "A"
    elif txt[-1] < 'Z':
        return txt[:-1] + chr(ord(txt[-1]) + 1)
    else:
        return _str_incr(txt[:-1]) + "0"


class Cross(QObject):
    """pyqtgraph Cross that migh follows mouse."""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self._parent = parent
        self._vLine = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("w")
        )
        self._hLine = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen("w")
        )
        self._parent.addItem(self._vLine, ignoreBounds=True)
        self._parent.addItem(self._hLine, ignoreBounds=True)
        self.hide()
        self._fixed = False
        # Update crosshair position with mouse movement
        # self._proxy = pg.SignalProxy(self._parent.scene().sigMouseMoved,
        #                              rateLimit=60, slot=self._mouse_moved)

    def _mouse_moved(self, event: tuple):
        pos = event  # QPointF
        if self._parent.sceneBoundingRect().contains(pos):
            mouse_point = self._parent.vb.mapSceneToView(pos)
            self._vLine.setPos(mouse_point.x())
            self._hLine.setPos(mouse_point.y())

    def show(self):
        self._vLine.show()
        self._hLine.show()
        self._visible = True

    def hide(self):
        self._vLine.hide()
        self._hLine.hide()
        self._visible = False

    def freeze(self):
        self._parent.scene().sigMouseMoved.disconnect(self._mouse_moved)

    def thaw(self):
        self._parent.scene().sigMouseMoved.connect(self._mouse_moved)


class ScanImage(QObject):
    """Group of pyqtgraph objects used to display an image."""

    # exposed members
    ROI: BoundedROI = None

    def __init__(self, glw: pg.GraphicsLayoutWidget, idx, MAX_ROWS: int = 2):
        super().__init__(glw)  # set parent so we get signals
        row = idx % MAX_ROWS
        col = idx // MAX_ROWS
        pi = glw.addPlot(row, col)
        pi.invertY(True)
        self._plot_item = pi

        pi.setAspectLocked(True)
        pi.setMouseEnabled(x=True, y=True)  # No shifting allowed
        pi.setLabel("left", "Y", units="µm")
        pi.setLabel("bottom", "X", units="µm")
        # Create ImageItem
        ii = pg.ImageItem()
        self._image_item = ii
        pi.addItem(ii)
        self._guideline = pg.InfiniteLine(
            pos=0,
            angle=0,
            movable=False,
            pen={"color": "r", "width": 1},
        )
        self._guideline.hide()
        pi.addItem(self._guideline)
        self.ROI = BoundedROI(
            (
                0,
                0,
            ),
            # aspectLocked=True,
        )
        self.ROI.addScaleHandle(
            (
                0,
                0,
            ),
            (
                1.0,
                1.0,
            ),
        )
        # self.ROI.addScaleHandle(
        #     (
        #         1,
        #         1,
        #     ),
        #     (
        #         0.0,
        #         0.0,
        #     ),
        # )
        pi.addItem(self.ROI)
        self.ROI.hide()
        self._cross = Cross(pi)
        self._cross.show()

    def enable_scan(self, params: scan_parameters.RegionScanParameters):
        self._params = params
        self._guideline.show()

    def disable_scan(self, *args):
        self._guideline.hide()

    def show_ROI(self):
        self.ROI.show()

    def hide_ROI(self):
        self.ROI.hide()

    def update_guide_line(self, line_number: int):
        """
        TODO: set parameters befor scanning.
        """
        self._guideline.setValue(
            line_number * self._params.pixel_size + self._params.start_point[1]
        )

    def update_image(self, image: np.ndarray):
        """Update image."""
        self._image_item.setImage(image, autoLevels=True)

    # from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent as mce
    def _handle_click(self, event):
        # Get mouse click position in scene coordinates
        self._click_callback(self._plot_item.vb.mapToView(event.pos()))

    def enable_cross(self, callback):
        self._click_callback = callback
        self._cross.thaw()
        self._cross.show()
        self._plot_item.scene().sigMouseClicked.connect(
            self._handle_click,
        )

    def disable_cross(self):
        self._cross.freeze()
        self._cross.hide()
        try:
            self._plot_item.scene().sigMouseClicked.disconnect(
                self._handle_click,
            )
        except Exception as e:
            print("Excception disabling creoss", e, type(e))

    def __del__(self):
        # self._plot_item.removeItem(self.ROI)
        # self._plot_item.removeItem(self._guideline)
        # self._plot_item.removeItem(self._cross)
        del self.ROI
        del self._guideline
        s = super()
        if hasattr(s, "__del__"):
            s.__del__(self)


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

    _guide_line: pg.InfiniteLine = None
    _scan_params: scan_parameters.RegionScanParameters
    _center: np.ndarray  # 3D Center position of scan
    _ROI: BoundedROI = None
    imagen: np.ndarray = None  # Image scan data array
    _must_restart = (
        False  # Flas we asked the scanner to stop so we can restart
    )
    _scan_modes: dict[str, base_scan.ScanModeInfo]
    _save_fd = None

    def __init__(
            self,
            scanner: base_scan.BaseScan,
            fname_s: FileNameServer | None = None,
            *args, **kwargs):
        """Init FrontEnd window.

        Parameters
        ----------
        scanner_class : class
            Scanning backend class
        """
        super().__init__(*args, **kwargs)

        self._fname_s = fname_s or FileNameServer()

        self.scanner: base_scan.BaseScan = scanner
        scanner.register_callbacks(
            self.handle_scan_start, self._emit_end, self._emit_linescan
        )
        self.single_scan_mode = False

        # Initialize attributes
        self.setWindowTitle("Main Window with Scanner Plot")
        self.setGeometry(100, 100, 1000, 800)
        self.scan_region_window = None
        self.is_scanning = False

        # Data saving attributes
        self.frames = []
        self.last_frame = None

        # Scanning parameters
        self._scan_params = None
        self.process_scanner_modes()
        # Center position (x, y, z)
        self._center = np.array(
            (
                10.0,
                10.0,
                3.0,
            ),
            dtype=np.float64,
        )

        # Create dockable windows and UI components
        self._scan_images: list[ScanImage] = []
        self.create_dock_widgets()
        # extras

        # Update parameters and image extents
        self.update_parameters()
        self.update_image_extents()
        self._update_pixel_size()

        # Connect signals
        self.line_arrived_signal.connect(self.receive_line)
        self.scan_end_signal.connect(self.handle_scan_end)
        self.scan_start_signal.connect(self.handle_scan_start)

        self._map_window = MapWindow((0, 20, 0, 20), 0.02, parent=self)
        self._map_window.show()

    def process_scanner_modes(self):
        self._scan_modes = {
            mode.description: mode for mode in self.scanner.get_scan_modes()
        }

    def _get_selected_scan_mode(self) -> base_scan.ScanModeInfo:
        return self._scan_modes[self._scanmode_db.currentText()]

    def _create_scan_images(self):
        """Recrerate all displayed images according to new mode."""
        sm = self._get_selected_scan_mode()
        self.graphics_layout.clear()
        # Borrar previo
        for si in self._scan_images:
            del si
        self._scan_images: list[ScanImage] = []
        for idx in range(sm.images_per_frame):
            self._scan_images.append(ScanImage(self.graphics_layout, idx))
        self._ROI = self._scan_images[0].ROI  # one === all
        self._scan_images[0].ROI.sigRegionChanged.connect(self.roi_changed)

    # ----------------------------
    # Dockable windows
    # ----------------------------
    def create_dock_widgets(self):
        """Creates dockable widgets for main window.

        Includes: controls and plot area, parameters dock, 1D scan dock,
        positioner dock, and data saving options.
        """
        # Create the dockable windows
        self.create_parameters_dock()
        self.update_parameters()
        self.create_positioner_dock()
        self.create_file_dock()
        # Dock widget for controls and plot
        self.controls_plot_dock = QDockWidget("2D Scan", self)
        self.controls_plot_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
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
        self.start_stop_button.setSizePolicy(
            QSizePolicy.Fixed, QSizePolicy.Fixed
        )
        buttons_layout.addWidget(self.start_stop_button)

        # ROI toggle button
        self.roi_button = QPushButton("ROI")
        self.roi_button.setCheckable(True)
        self.roi_button.clicked.connect(self.toggle_roi)
        self.roi_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        buttons_layout.addWidget(self.roi_button)

        # Create a PyQtGraph GraphicsLayoutWidget
        self.graphics_layout = pg.GraphicsLayoutWidget()
        controls_plot_layout.addWidget(self.graphics_layout)

        # Create a PlotItem with axes
        # self.plot_item = self.graphics_layout.addPlot()
        # self.plot_item.setAspectLocked(True)
        # self.plot_item.setMouseEnabled(x=True, y=True)

        # Set axes labels
        # self.plot_item.setLabel("left", "Y", units="µm")
        # self.plot_item.setLabel("bottom", "X", units="µm")

        # Create ImageItem
        # self.image_item = pg.ImageItem()
        # self.plot_item.addItem(self.image_item)
        # self.plot_item.setMouseEnabled(x=False, y=False)  # No shifting
        # self._guide_line = pg.InfiniteLine(pos=0, angle=0, movable=False,
        #                                    pen={'color': 'r', 'width': 1},)
        # self._guide_line.hide()
        # self._ROI = BoundedROI((0, 0,))
        # self._ROI.addScaleHandle((0, 0,), (1., 1.,))
        # self._ROI.addScaleHandle((1, 1,), (0., 0.,))
        # self._ROI.hide()
        # # self._ROI.setZValue(10)
        # self._ROI.sigRegionChanged.connect(self._update_map_roi)
        # # TODO: remove this, used for developing. Updates ROI *on the fly*
        # self._ROI.sigRegionChangeFinished.connect(self.roi_changed)
        # self.plot_item.addItem(self._guide_line)
        # self.plot_item.addItem(self._ROI)
        # self._cross = Cross(self.plot_item)
        # self._cross.hide()
        self._create_scan_images()
        self.update_roi(self._scan_params.center, self._scan_params.line_length_fast)

        # # Create histogram for color scaling
        # self.histogram = pg.HistogramLUTItem()
        # self.histogram.setImageItem(self.image_item)
        # self.graphics_layout.addItem(self.histogram)

    def create_parameters_dock(self):
        """
        Creates the dockable window with scan parameter settings.

        Includes: inputs for scan type, ranges, dwell time, acceleration, and
        number of pixels.
        """
        dock = QDockWidget("Set Parameters", self)
        self.parameters_dock = dock
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        dock_widget = QWidget()
        dock.setWidget(dock_widget)

        grid_layout = QGridLayout()
        dock_widget.setLayout(grid_layout)

        # 2D Scan type selection
        # self.scan_type_widget = QWidget()
        scan_type_layout = QHBoxLayout()
        # self.scan_type_widget.setLayout(scan_type_layout)
        self.scan_type_group = QButtonGroup()
        self._radiobuttons: dict[
            QRadioButton, scan_parameters.RegionScanType
        ] = {}
        for st in scan_parameters.RegionScanType:
            nrb = QRadioButton(st.value)
            self._radiobuttons[nrb] = st
            self.scan_type_group.addButton(nrb)
            scan_type_layout.addWidget(nrb)
        list(self._radiobuttons.keys())[0].setChecked(True)
        self.scan_type_group.buttonToggled.connect(self._on_scan_type_change)

        grid_layout.addWidget(QLabel("2D Scan Type:"), 0, 0)
        grid_layout.addLayout(scan_type_layout, 0, 1, 1, -1)
        # grid_layout.addWidget(self.scan_type_widget, 0, 1)

        # X Range input
        self.x_range_input = pyqtutils.create_float_lineedit(
            5
        )  # Default X range value
        grid_layout.addWidget(QLabel("Fast Range (μm):"), 1, 0)
        grid_layout.addWidget(self.x_range_input, 1, 1)

        # Y Range input
        self.y_range_input = pyqtutils.create_float_lineedit(
            5
        )  # Default Y range value
        grid_layout.addWidget(QLabel("Slow Range (μm):"), 2, 0)
        grid_layout.addWidget(self.y_range_input, 2, 1)

        # Dwell time input
        self.dwell_time_input = pyqtutils.create_float_lineedit(
            400
        )  # Default dwell time
        grid_layout.addWidget(QLabel("Dwell Time (µs):"), 3, 0)
        grid_layout.addWidget(self.dwell_time_input, 3, 1)

        # Acceleration Input
        self.a_aux_input = pyqtutils.create_float_lineedit(
            "0.1"
        )  # Default a_aux
        grid_layout.addWidget(QLabel("Acceleration (µm/ms²):"), 4, 0)
        grid_layout.addWidget(self.a_aux_input, 4, 1)

        # Number of pixels input
        self.num_pixels_input = pyqtutils.create_int_lineedit(
            "40"
        )  # Default number of pixels
        grid_layout.addWidget(QLabel("Number of Pixels:"), 5, 0)
        grid_layout.addWidget(self.num_pixels_input, 5, 1)

        # Pixel size display (read-only)
        pixel_size_label = QLabel("Pixel Size (µm):")
        pixel_size_label.setStyleSheet(
            "QLabel { font-weight: bold; }"
        )  # Optional styling
        self.pixel_size_display = QLabel("")  # will be updated later
        dock_background_color = (
            self.palette().color(self.backgroundRole()).name()
        )
        self.pixel_size_display.setStyleSheet(
            f"QLabel {{ background-color : {dock_background_color}; }}"
        )

        # Add pixel size label and display in the grid
        grid_layout.addWidget(pixel_size_label, 1, 2)
        grid_layout.addWidget(self.pixel_size_display, 1, 3)

        # Reset parameters button
        self.reset_params_button = QPushButton("Init Params")
        self.reset_params_button.clicked.connect(
            self.reset_parameters_to_initial
        )
        grid_layout.addWidget(self.reset_params_button, 2, 2)

        # Apply ROI selection button
        self.apply_roi_button = QPushButton("Apply ROI")
        self.apply_roi_button.clicked.connect(self._set_roi_from_params)
        grid_layout.addWidget(self.apply_roi_button, 3, 2)

        # Connect textChanged signals -> unckeck save scan data checkbox
        self.x_range_input.textChanged.connect(self.changed_px_size)
        self.y_range_input.textChanged.connect(self.changed_px_size)
        self.num_pixels_input.textChanged.connect(self.changed_px_size)

        # scan modes and detector
        self._detector_db = QComboBox(self)
        self._detector_db.addItems(self.scanner.get_detectors())
        grid_layout.addWidget(QLabel("Detector:"), 6, 0)
        grid_layout.addWidget(self._detector_db, 6, 1)

        self._scanmode_db = QComboBox(self)
        self._scanmode_db.addItems([_ for _ in self._scan_modes])
        grid_layout.addWidget(QLabel("Scan mode:"), 7, 0)
        grid_layout.addWidget(self._scanmode_db, 7, 1)
        self._scanmode_db.currentTextChanged.connect(self._on_scanmode_change)

        grid_layout.setRowStretch(8, 1)

        # Store initial parameters for resetting
        self._initial_params = {
            "scan_type": scan_parameters.RegionScanType.XY.value,
            "x_range": self.x_range_input.text(),
            "y_range": self.y_range_input.text(),
            "dwell_time": self.dwell_time_input.text(),
            "acceleration": self.a_aux_input.text(),
            "num_pixels": self.num_pixels_input.text(),
        }

    def create_positioner_dock(self):
        """Creates dockable window for positioner controls.

        Includes: controls for single scan, choosing center, full scan option,
        and moving to a specific center.
        """
        dock = QDockWidget("Scan Data and Position Paramters", self)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        dock_widget = QWidget()
        dock.setWidget(dock_widget)

        layout = QVBoxLayout()
        dock_widget.setLayout(layout)

        # New buttons arranged side by side
        buttons_layout = QHBoxLayout()
        self.single_scan_button = QPushButton("Single Scan")
        self.choose_center_button = QPushButton("Choose Center")
        self.choose_center_button.setCheckable(True)
        self.full_scan_checkbox = QCheckBox("Full Scan")

        # Add buttons to layout
        buttons_layout.addWidget(self.single_scan_button)
        # buttons_layout.addWidget(self.scan_line_button)
        buttons_layout.addWidget(self.choose_center_button)
        buttons_layout.addWidget(self.full_scan_checkbox)
        layout.addLayout(buttons_layout)

        # Position
        self._position_inputs: list[QLineEdit] = [
            pyqtutils.create_float_lineedit(f"{d:.3f}") for d in self._center
        ]
        pos_layout = QHBoxLayout()
        for le, name in zip(self._position_inputs, ["X", "Y", "Z"]):
            pos_layout.addWidget(QLabel(name), stretch=0)
            pos_layout.addWidget(le, stretch=1)
        self.move_to_button = QPushButton("Move to")

        layout.addWidget(QLabel("Central position (µm):"))
        layout.addLayout(pos_layout)
        layout.addWidget(self.move_to_button)
        layout.addStretch(1)
        # Connect buttons to functions
        self.single_scan_button.clicked.connect(self.single_scan)
        self.choose_center_button.clicked.connect(self.choose_center)

        self.move_to_button.clicked.connect(self.move_to_position)

    def create_file_dock(self):
        """Creates dockable window for saving data."""
        dock = QDockWidget("Data saving parameters", self)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock_widget = QWidget()
        dock.setWidget(dock_widget)
        layout = QVBoxLayout()
        dock_widget.setLayout(layout)

        # New buttons arranged side by side
        # TODO: group this into a helper function
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Directory"), stretch=0)
        self._dir_le = QLineEdit(str(self._fname_s.get_base_dir()))
        dir_layout.addWidget(self._dir_le)
        self._choose_dir_btn = QPushButton("Choose")
        dir_layout.addWidget(self._choose_dir_btn)

        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Base filename:"), stretch=0)
        self._fname_le = QLineEdit("scan")
        name_layout.addWidget(self._fname_le)
        name_layout.addWidget(QLabel("suffix:"), stretch=0)
        self._sufix_le = QLineEdit("00")
        name_layout.addWidget(self._sufix_le)
        # self._choose_name_btn = QPushButton("Choose")
        # name_layout.addWidget(self._choose_name_btn)
        # Checkbox auto_append_date
        # Checkbox auto_increment (y poner nro)

        save_layout = QHBoxLayout()
        # save_layout.addWidget(QLabel("Save data?"), stretch=0)
        self._save_chk = QCheckBox("Save_data")
        save_layout.addWidget(self._save_chk)

        layout.addLayout(dir_layout)
        layout.addLayout(name_layout)
        layout.addLayout(save_layout)
        layout.addStretch(1)
        # Connect buttons to functions
        # self.single_scan_button.clicked.connect(self.single_scan)
        self._choose_dir_btn.clicked.connect(self._choose_dir)
        self._filesave_dock = dock

    def changed_px_size(self):
        """Uncheck -> Save Scan Data checkbox when pixel size changes."""
        ...

    # ----------------------------
    # Changing/Updating Data and Parameters
    # ----------------------------
    def _get_selected_scan_type(self) -> scan_parameters.RegionScanType:
        """Return selected scan type."""
        checked_button = self.scan_type_group.checkedButton()
        return self._radiobuttons[checked_button]

    @pyqtSlot(str)
    def _on_scanmode_change(self, value: str):
        """Depende el tipo de escaneo."""
        self.update_parameters()
        old_scan_params = self._scan_params
        self._create_scan_images()  # Esto resetea los parámetros al recrear los ROIs
        self._set_scan_params(old_scan_params)  # Suci sucio
        self.update_roi(self._scan_params.center, self._scan_params.line_length_fast)

    @pyqtSlot(QAbstractButton, bool)
    def _on_scan_type_change(self, button: QRadioButton, checked: bool):
        """XY, XZ, YZ."""
        # print("el botón", button, "ahora está", checked)
        if not checked:
            return
        scan_type = self._get_selected_scan_type()  # podríamos usar button
        if self._scan_params is not scan_type:
            # TODO: recenter ROI and update mapwindow -> el ROI tiene medio pixel de mas
            self.update_parameters()
            logger.debug("Update por update ROI cn %s %s", self._scan_params.center, self._scan_params.line_length_fast)
            self.update_roi(self._scan_params.center, self._scan_params.line_length_fast)
            self._reset_images()
            self.update_image_extents()
            self._map_window.reset_contents()

    def update_parameters(self):
        """Collects parameters from the UI and updates the info."""
        scan_type = self._get_selected_scan_type()

        # Get X and Y ranges (fast/slow)
        x_range_value = float(self.x_range_input.text())
        y_range_value = float(self.y_range_input.text())
        # FIXME: non-square

        # Define start and end points based on scan type
        center = self._center[_scantype_axes_map[scan_type]]
        logger.debug("Center: %s", center)

        shift = np.array((x_range_value, y_range_value)) / 2
        start_point = center - shift
        end_point = center + shift

        dwell_time = float(self.dwell_time_input.text())
        num_pixels = int(self.num_pixels_input.text())
        a_aux = self.a_aux_input.text()

        if (
            (not num_pixels)
            or (x_range_value < 1e-3)
            or (x_range_value < 1e-3)
        ):
            logger.error("Valores muy pequenos")
            raise ValueError("Range not allowed")

        new_scan_params = scan_parameters.RegionScanParameters(
            scan_type=scan_type,
            start_point=start_point,
            end_point=end_point,
            center=center,
            dwell_time=dwell_time,
            # full_data=self.full_scan_checkbox.isChecked(),
            true_px_num=num_pixels,
            acceleration=0,  # FIXME
            a_aux=a_aux,
        )
        self._scan_params = new_scan_params
        self._update_pixel_size()

    def _update_pixel_size(self) -> float:
        """Calculate the pixel size based on fast range and update display."""
        # TODO: manage pixel size for non-square scan regions
        # FIXME: Non-square
        s_range = self._scan_params.end_point - self._scan_params.start_point
        num_pixels = self._scan_params.true_px_num
        pixel_size = s_range[0] / num_pixels if num_pixels else np.inf
        self.pixel_size_display.setText(f"{pixel_size:.4f}")
        return pixel_size

    def _set_scan_params(self, params: scan_parameters.RegionScanParameters):
        """Sets GUI params based on RegionScanParameters."""
        scan_type = self._get_selected_scan_type()
        self._center[_scantype_axes_map[scan_type]] = params.center
        x_range = params.end_point[0] - params.start_point[0]
        y_range = params.end_point[1] - params.start_point[1]
        self.x_range_input.setText(f"{x_range:.3f}")
        self.y_range_input.setText(f"{y_range:.3f}")
        self.dwell_time_input.setText(f"{params.dwell_time:.3f}")
        self.a_aux_input.setText(f"{params.acceleration:.3f}")
        self.num_pixels_input.setText(f"{params.true_px_num}")

        for le, coord in zip(self._position_inputs, self._center):
            le.setText(f"{coord:.3f}")
        self.update_parameters()

    def reset_parameters_to_initial(self):
        """Resets the parameters to their initial values."""
        # Reset UI inputs to init values
        self.x_range_input.setText(self._initial_params["x_range"])
        self.y_range_input.setText(self._initial_params["y_range"])
        self.dwell_time_input.setText(self._initial_params["dwell_time"])
        self.a_aux_input.setText(self._initial_params["acceleration"])
        self.num_pixels_input.setText(self._initial_params["num_pixels"])

        # Reset center position
        self._center = np.array((10.0, 10.0, 3.0), dtype=np.float64)
        for le, coord in zip(self._position_inputs, self._center):
            le.setText(f"{coord:.3f}")

        self.update_parameters()

    def move_to_position(self):
        """Sets the center position according to text boxes."""
        new_center = [float(le.text()) for le in self._position_inputs]
        logger.info("Setting new center position: %s", new_center)
        self._center[:] = new_center
        # FIXME: Move piezo
        # piezo.move_to(new_center)
        self.update_parameters()
        self.update_roi(self._scan_params.center, self._scan_params.line_length_fast)

    def update_image_extents(self):
        """Updates extents and pixel size of image before scanning.

        TODO: recenter
        TODO: Check half-pixels
        """
        px_s = self._scan_params.pixel_size
        sp = self._scan_params.start_point
        tr = QtGui.QTransform()
        tr.scale(px_s, px_s).translate(
            # sp[0] / px_s - 0.5, sp[1] / px_s - 0.5
            sp[0] / px_s, sp[1] / px_s,
        )
        for si in self._scan_images:
            si._image_item.setTransform(
                tr
            )

    # ----------------------------
    # Data Handling/Receiving
    # ----------------------------
    @pyqtSlot()
    def handle_scan_end(self):
        """Manages scan end and updates UI accordingly."""
        logger.info("Scan end")
        if self._save_fd:
            self._save_fd.close()
            self._save_fd = None
            if self._sufix_le.text().strip():
                self._sufix_le.setText(_str_incr(self._sufix_le.text()))
        self.is_scanning = False
        self.single_scan_mode = False
        self.update_scan_button_label(False)
        self.single_scan_button.setEnabled(True)
        for si in self._scan_images:
            si.disable_scan()
        self.parameters_dock.setEnabled(True)
        self._filesave_dock.setEnabled(True)
        self.scanner.cleanup_scan()

    # @pyqtSlot()
    def handle_scan_start(
        self, params: scan_parameters.RegionScanParameters, shape: np.ndarray, n_layers: int
    ):  # scan_parameters,
        """Manages scan end and updates UI accordingly."""
        logger.info("Scan start")
        logger.debug("Data layers: %s", n_layers)
        logger.debug("Data shape: %s", shape)
        self.parameters_dock.setEnabled(False)
        self._filesave_dock.setEnabled(False)
        self.is_scanning = True
        self.update_scan_button_label(True)
        self.single_scan_button.setEnabled(False)

        if self._save_chk.isChecked():
            p_file_name = self._fname_s.get_base_dir() / self._fname_s.get_base_name()
            p_file_name = p_file_name.with_name(p_file_name.stem + "_params").with_suffix(".json")
            params.save_to(p_file_name)
            fname = self._fname_s.get_base_dir() / self._fname_s.get_base_name()
            fname = fname.with_name(fname.stem + "_scan").with_suffix(".NPY")
            self._save_fd = open(fname, "xb")

        self.imagen = np.zeros(
            [n_layers, *shape]
        )  # TODO: add layer and images
        self.update_image_extents()
        self.last_frame = None
        for si in self._scan_images:
            si.enable_scan(self._scan_params)

    def start_stop_scan(self):
        """Starts or stops the scan based on the toggle button state."""
        self.update_parameters()
        if self.is_scanning:
            self.scanner.stop_scan()
        else:
            try:  # FIXME: send scan mode?
                self._publish_filename()
                self.scanner.start_scan(
                    self._scan_params, self._scanmode_db.currentText()
                )
            except Exception as e:
                logger.error("Error starting scan: %s (%s)", e, type(e))

    def _emit_linescan(self, line_data: np.ndarray, lineno: int, last: bool):
        """Proxy callback to signal line data.

        Parameters
        ----------
            line_data : np.ndarray
               Data of current line
            lineno : int
                Line number in scan
        """
        self.line_arrived_signal.emit(line_data, lineno)
        if (
            self.single_scan_mode
            and lineno >= self._scan_params.true_px_num - 1
        ):
            print("Fin single scan")
            return True
        return False  # do not stop scanning

    def _emit_end(self):
        """Proxy callback to signal scanning has ended."""
        logger.debug("Emiting end signal")
        self.scan_end_signal.emit()

    @pyqtSlot(np.ndarray, int)
    def receive_line(self, line_data: np.ndarray, line_number: int):
        """
        Receives a line of data from the scanning backend and updates image.

        Parameters
        ----------
            line_data : np.ndarray
                Data of current line
            line_number : int
                Line number in scan
        """
        # TODO: chequeos: scanning, limites, etc.
        self.imagen[:, :, line_number] = line_data
        for si, img in zip(self._scan_images, self.imagen):
            si.update_image(img)
            si.update_guide_line(line_number)
        # self.histogram.imageChanged(autoLevel=True)
        # self.update_guide_line(line_number)

        # Check if frame is complete
        if line_number == self._scan_params.true_px_num - 1:
            # Make a copy of the current frame
            # FIXME: multiples imágenes
            frame = np.copy(self.imagen[0])
            self.last_frame = frame  # Store the last completed frame
            self._map_window.add_region(
                frame,
                self._scan_params.line_length_fast,
                self._scan_params.center,
                self._scan_params.dwell_time,
            )

            # Save data if requested
            if self._save_fd:
                np.save(self._save_fd, self.imagen)

        if (
            self.single_scan_mode
            and line_number >= self._scan_params.true_px_num - 1
        ):
            self.start_stop_scan()

    # ----------------------------
    # ROI
    # ----------------------------
    @pyqtSlot(bool)
    def toggle_roi(self, checked: bool):
        """Toggles visibility of ROI on plot.

        Parameters
        ----------
            checked : bool
        """
        # FIXME: multiples imágenes
        if checked:
            for si in self._scan_images:
                si.show_ROI()
        else:
            for si in self._scan_images:
                si.hide_ROI()

    @pyqtSlot(object)
    def roi_changed(self, roi: object):
        """(For debugging only)

        Update scan parameters based on ROI

        Parameters
        ----------
            roi : object
        """
        # FIXME: multiples imágenes
        self.apply_ROI_selection(True)

    def _set_roi_from_params(self):
        x_range = float(self.x_range_input.text())
        y_range = float(self.y_range_input.text())
        if x_range != y_range:
            logger.error("Fast / slow range mismatch (%s/%s)", x_range, y_range)
            self.y_range_input.setText(self.x_range_input.text())
        self.update_parameters()
        self.update_roi(self._scan_params.center, self._scan_params.line_length_fast)

    def update_roi(self, center: tuple, data_range: float):
        """Updates ROI position based on data."""
        self._ROI.setSize((data_range, data_range), update=False, finish=False)
        self._ROI.setPos(
            center[0] - data_range / 2,
            center[1] - data_range / 2,
            update=True,  # Causes one time looping
        )

    @pyqtSlot(bool)
    def apply_ROI_selection(self, checked: bool):
        """Apply ROI selection.

        Be aware that the axes depend on the last scan type performed
        # FIXME: W?

        Parameters
        ----------
            checked : bool (needed only for signal compatibility).
        """
        roi_pos = self._ROI.pos()
        roi_size = self._ROI.size()
        logger.debug("Aplicando ROI %s, %s", roi_pos, roi_size)
        # FIXME: manage half pixels
        f_start = roi_pos.x()
        f_end = f_start + roi_size.x()
        s_start = roi_pos.y()
        s_end = s_start + roi_size.y()
        # Update center position
        axes = _scantype_axes_map[self._scan_params.scan_type]
        self._center[axes[0]] = (f_start + f_end) / 2
        self._center[axes[1]] = (s_start + s_end) / 2

        # Update text
        self._position_inputs[axes[0]].setText(f"{self._center[axes[0]]:.3f}")
        self._position_inputs[axes[1]].setText(f"{self._center[axes[1]]:.3f}")
        # Update range inputs
        self.x_range_input.setText(f"{roi_size.x():.3f}")
        self.y_range_input.setText(f"{roi_size.y():.3f}")
        # DO NOT Update parameters
        # self.update_parameters()
        self.request_restart()

    def request_restart(self):
        """Crops region selected by ROI and restarts.

        FIXME: Change behaviour or comment.
        """
        if self.is_scanning:
            self._must_restart
            self.scanner.stop_scan()
            # self.update_parameters()
            # self.start_stop_scan()

    # ----------------------------
    # Types of Scanning Options
    # ----------------------------
    def single_scan(self):
        """Start a single scan."""
        self.single_scan_mode = True
        self.single_scan_button.setEnabled(
            False
        )  # Disable button (avoid chaos)
        # self.start_scan()
        self.start_stop_scan()
        # print("Single Scan button pressed")

    def update_guide_line(self, line_number: int):
        """Updates position of 'current position' guide line.

        Parameters
        ----------
        line_number : int
            Current scan line number
        """
        self._guide_line.setValue(
            line_number * self._scan_params.pixel_size
            + self._scan_params.start_point[1]
        )

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

    def choose_center(self, checked):
        """Toggles choose center mode.

        Parameters
        ----------
            checked : bool
        """
        if checked:
            self.choose_center_mode = True
            for si in self._scan_images:
                si.enable_cross(self.plot_click)
            logger.debug("Choose Center mode activated")
        else:
            self.choose_center_mode = False
            try:
                for si in self._scan_images:
                    si.disable_cross()
            except TypeError as e:
                logger.error("typerror desconectando: %s", e)
                pass
            logger.debug("Choose Center mode deactivated")

    def plot_click(self, pos: QPointF):
        """Handles mouse click events on plot when choosing center.

        Parameters
        ----------
        post : QPointF
            x, y
        """
        if not self.choose_center_mode:
            return
        x, y = pos.x(), pos.y()
        logger.info("New center selected at (%.3f, %.3f)", x, y)
        # Update central position
        axes = _scantype_axes_map[self._scan_params.scan_type]
        self._center[axes[0]] = x
        self._center[axes[1]] = y
        # Update the position inputs
        self._position_inputs[axes[0]].setText(f"{x:.3f}")
        self._position_inputs[axes[1]].setText(f"{y:.3f}")

        self.update_parameters()
        self.choose_center_button.setChecked(False)
        self.choose_center(False)
        self.choose_center_mode = False

        self.update_roi(self._scan_params.center, self._scan_params.line_length_fast)

    def _reset_images(self):
        """Clear all images."""
        if getattr(self, 'imagen', None) is not None:
            self.imagen[:] = 0
            for si, img in zip(self._scan_images, self.imagen):
                si.update_image(img)

    def _publish_filename(self):
        """Publish base finlename + suffix to filename server."""
        fname = self._fname_le.text()
        if (suffix := self._sufix_le.text().strip().upper()):
            fname += "_" + suffix
        directory = self._dir_le.text()
        # print(directory, fname)
        self._fname_s.set_base_dir(directory)
        self._fname_s.set_base_name(fname)
        # print(self._fname_s.get_base_dir() / self._fname_s.get_base_name())

    def _choose_dir(self, checked: bool):
        """Show dir seleciton dialog."""
        dirname = QFileDialog.getExistingDirectory(self, directory=self._dir_le.text(), options=QFileDialog.Option.ShowDirsOnly)
        if dirname:
            self._dir_le.setText(dirname)

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
            logger.debug("Stopping scan before exiting")
            self.scanner.stop_scan()
        # self.close_files()
        self._map_window.close()
        super().closeEvent(event)


if __name__ == "__main__":
    scanner = mock_scanner()
    app = QApplication(sys.argv)
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    # window = FrontEnd(NIDAQScanner_Manager.NIDAQScanner_2DSignals)
    window = FrontEnd(scanner)

    window.show()
    window.raise_()
    window.activateWindow()

    rv = app.exec_()
    # app.exec_()
    app.quit()
    sys.exit(rv)
