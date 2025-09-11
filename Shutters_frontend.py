# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 11:05:35 2025

@author: Luis1
"""
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLabel, QHBoxLayout, QButtonGroup, QRadioButton, QFrame
)
from PyQt5.QtCore import Qt
import importlib
import Shutters_backend
importlib.reload(Shutters_backend)
from mock_shutters import mock_shutters
from Shutters_backend import NIDAQShuttersBackend

class FrontEnd(QMainWindow):
    def __init__(self, shutters):
        super().__init__()

        self.setWindowTitle("Shutter Control Panel")
        self.setGeometry(200, 200, 900, 600)

        # Backend de shutters (puede ser mock_shutters o el real)
        self.shutters = shutters
        self.shutter_groups = []

        #  Estilo oscuro elegante
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
            QFrame {
                background-color: #1E1E1E;
                border-radius: 10px;
                border: 1px solid #333333;
            }
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #eeeeee;
            }
            QRadioButton {
                font-size: 14px;
                font-weight: bold;
                padding: 6px 12px;
                border-radius: 12px;
                border: 2px solid #666666;
                color: white;
            }
            QRadioButton::indicator { width: 0px; height: 0px; }  /* Ocultar circulito clásico */

            /* Colores activos */
            QRadioButton#off:checked { background-color: #e53935; border: 2px solid #ff6f60; }
            QRadioButton#on:checked { background-color: #43a047; border: 2px solid #66bb6a; }
            QRadioButton#auto:checked { background-color: #1e88e5; border: 2px solid #64b5f6; }

            /* Colores inactivos */
            QRadioButton#off:!checked { background-color: #3d1f1f; color: #aaaaaa; }
            QRadioButton#on:!checked { background-color: #1b4020; color: #aaaaaa; }
            QRadioButton#auto:!checked { background-color: #1a2f4a; color: #aaaaaa; }
        """)

        # Layout central con 4 paneles iguales
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Crear los shutters dinámicamente desde el backend
        for shutter_id in sorted(self.shutters.shutter_states.keys()):
            shutter_panel = self.create_shutter_panel(shutter_id)
            main_layout.addWidget(shutter_panel, 1)  # "1" = proporción igual

        self.setCentralWidget(central_widget)

    def create_shutter_panel(self, shutter_id):
        """Crea un panel con controles de un shutter."""
        frame = QFrame()
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(20)

        label = QLabel(f"Shutter {shutter_id+1}")

        # Botones de estado
        btn_off = QRadioButton("Off"); btn_off.setObjectName("off")
        btn_on = QRadioButton("On"); btn_on.setObjectName("on")
        btn_auto = QRadioButton("Auto"); btn_auto.setObjectName("auto")

        # Agruparlos
        group = QButtonGroup(self)
        group.addButton(btn_off)
        group.addButton(btn_on)
        group.addButton(btn_auto)

        # Estado inicial desde backend
        state = self.shutters.shutter_states[shutter_id]
        if state == "off":
            btn_off.setChecked(True)
        elif state == "on":
            btn_on.setChecked(True)
        else:
            btn_auto.setChecked(True)

        # Conectar evento
        group.buttonClicked.connect(
            lambda btn, s_id=shutter_id: self.change_shutter_state(s_id, btn.objectName())
        )

        # Armar layout
        layout.addWidget(label)
        layout.addStretch()
        layout.addWidget(btn_off)
        layout.addWidget(btn_on)
        layout.addWidget(btn_auto)

        # Guardar referencia
        self.shutter_groups.append(group)

        return frame

    def change_shutter_state(self, shutter_id, new_state):
        """Actualiza el estado de un shutter en el backend."""
        self.shutters.shutter_states[shutter_id] = new_state
        print(f"Shutter {shutter_id+1} -> {new_state.upper()}")


if __name__ == "__main__":
    #shutters = mock_shutters(n_shutters=4)
    shutters = NIDAQShuttersBackend()

    app = QApplication(sys.argv)
    window = FrontEnd(shutters)
    window.show()
    sys.exit(app.exec_())


