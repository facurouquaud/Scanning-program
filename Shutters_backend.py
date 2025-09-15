# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 11:22:55 2025

@author: Luis1
"""

import nidaqmx
from typing import Callable, Optional, List, Tuple
import logging


StartCallback = Callable[[object, Tuple[int, int]], None]
EndCallback = Callable[[], None]

class ShuttersBackend:
    """Backend genérico para manejar el estado de shutters."""

    VALID_STATES = {"off", "on", "auto"}

    def __init__(self, n_shutters=4):
        # Estado inicial: todos apagados
        self.shutter_states = {i: "off" for i in range(n_shutters)}
        self._start_callbacks: List[StartCallback] = []
        self._end_callbacks: List[EndCallback] = []

    def set_state(self, shutter_id, state):
        """Cambia el estado de un shutter."""
        if state not in self.VALID_STATES:
            raise ValueError(f"Estado inválido: {state}")

        self.shutter_states[shutter_id] = state
        print(f"[BACKEND] Shutter {shutter_id+1} -> {state.upper()}")

    def get_state(self, shutter_id):
        return self.shutter_states[shutter_id]

    def reset_all(self):
        for s_id in self.shutter_states:
            self.shutter_states[s_id] = "off"
        print("[BACKEND] Todos los shutters apagados")


class NIDAQShuttersBackend(ShuttersBackend):
    def __init__(self, n_shutters=4, device="Dev1"):
        self._start_callbacks: List[StartCallback] = []
        self._end_callbacks: List[EndCallback] = []
        #  Importante: inicializar el padre para crear self.shutter_states
        super().__init__(n_shutters)

        self.device = device
        self.tasks = {}

        # Configurar un canal digital por cada shutter
        for s_id in range(n_shutters):
            line = f"{device}/port0/line{s_id}"
            task = nidaqmx.Task()
            task.do_channels.add_do_chan(line)
            self.tasks[s_id] = task

    def set_state(self, shutter_id, state):
        # Actualiza el diccionario en el padre
        super().set_state(shutter_id, state)

        # Traducción a señales TTL
        if state == "on":
            self.tasks[shutter_id].write(True)
        elif state == "off":
            self.tasks[shutter_id].write(False)
        elif state == "auto":
            # En "auto" quizás lo maneje el escaneo, no lo forzamos acá
            pass
    def register_callbacks(self,
                          shutter_start_callback: Optional[StartCallback] = None,
                          shutter_end_callback: Optional[EndCallback] = None
                         ):
        """Register scan callbacks."""
        if shutter_start_callback:
            self._start_callbacks.append(shutter_start_callback)
        if shutter_end_callback:
            self._end_callbacks.append(shutter_end_callback)
    def _execute_shutter_start_callbacks(self):
        for s_id, state in self.shutter_states.items():
            print(f"[DEBUG] Shutter {s_id+1} state={state}")
            if state == "auto":
                print(f"[DEBUG] Abrir shutter {s_id+1}")
                self.tasks[s_id].write(True)

        for callback in self._start_callbacks:
            try:
                callback()
            except Exception as e:
                logging.error(f"Start callback error: {e}")

    def _execute_shutter_end_callbacks(self):
        """Al terminar escaneo, cerrar los shutters en auto y notificar callbacks."""
        for s_id, state in self.shutter_states.items():
            if state == "auto":
                self.tasks[s_id].write(False)  # cerrar shutter
                print(f"[BACKEND] Shutter {s_id+1} (AUTO) -> OFF")

        for callback in self._end_callbacks:
            try:
                callback()
            except Exception as e:
                logging.error(f"Stop callback error: {e}")

    def close(self):
        for task in self.tasks.values():
            task.close()
