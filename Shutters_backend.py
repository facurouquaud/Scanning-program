# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 11:22:55 2025

@author: Luis1
"""

import nidaqmx

class ShuttersBackend:
    """Backend genérico para manejar el estado de shutters."""

    VALID_STATES = {"off", "on", "auto"}

    def __init__(self, n_shutters=4):
        # Estado inicial: todos apagados
        self.shutter_states = {i: "off" for i in range(n_shutters)}

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

    def close(self):
        for task in self.tasks.values():
            task.close()
