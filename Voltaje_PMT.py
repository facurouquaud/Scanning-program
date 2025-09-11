# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 11:25:44 2025

@author: Luis1
"""

"""
Rutina para controlar el voltaje que se entrega al PMT para comenzar a medir.
El voltaje tipico de medicion es 0.3V.

La placa programable que entrega el voltaje es una NIDAQ, desde su puerto ????.


El voltaje del NIDAQ esta montado sobre un offset de ???? V aprox. 
Para proteger el PMT se impone como maximo el valor 1.2V.

"""
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QSlider, QPushButton, QFrame
)
from PyQt5.QtCore import Qt
import nidaqmx

class PMTController(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initDAQ()

    def initDAQ(self):
        # Configuración de la tarjeta NIDAQ
        self.ao_channel = "Dev1/ao0"  # Cambiar según tu hardware
        self.ao_task = nidaqmx.Task()
        self.ao_task.ao_channels.add_ao_voltage_chan(
            self.ao_channel, min_val=0.0, max_val=10.0
        )
        self.ao_task.start()
    def initUI(self):
        self.setWindowTitle('Controlador Voltaje PMT')
        self.setGeometry(100, 100, 400, 200)

        # Layout principal
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)

        # Encabezado
        header = QLabel('Seleccione el voltaje del PMT')
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 16px; font-weight: bold;")
        main_layout.addWidget(header)

        # Info rango
        info = QLabel('Rango de trabajo: 0 - 1.25 V (usar ~0.3 V)')
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet("font-size: 12px; color: gray;")
        main_layout.addWidget(info)

        # Contenedor del slider
        slider_frame = QFrame()
        slider_layout = QVBoxLayout()
        slider_frame.setLayout(slider_layout)
        slider_frame.setStyleSheet("background-color: #f0f0f0; border-radius: 8px; padding: 10px;")

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1250)  # Para 0.001 V de resolución
        self.slider.setTickInterval(125)  # Marca cada 0.125 V
        self.slider.setSingleStep(1)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setValue(300)  # 0.3 V inicial
        self.slider.valueChanged.connect(self.update_label)
        slider_layout.addWidget(self.slider)

        # Label del valor seleccionado
        self.value_label = QLabel("Voltaje PMT = 0.300 V")
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        slider_layout.addWidget(self.value_label)

        main_layout.addWidget(slider_frame)

        # Botón enviar
        self.btn = QPushButton("Enviar a DAQ")
        self.btn.setStyleSheet("font-size: 14px; padding: 6px;")
        self.btn.clicked.connect(self.send_voltage)
        main_layout.addWidget(self.btn, alignment=Qt.AlignCenter)

        self.setLayout(main_layout)

    def update_label(self):
        n = self.slider.value() / 1000
        self.value_label.setText(f"Voltaje PMT = {n:.3f} V")

    def send_voltage(self):
        n = self.slider.value() / 1000
        print('Voltaje PMT =', n, 'V')
        self.ao_task.write(n)

    def closeEvent(self, event):
        try:
            self.ao_task.write(0)
            self.ao_task.close()
        except:
            pass
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PMTController()
    window.show()
    sys.exit(app.exec_())
