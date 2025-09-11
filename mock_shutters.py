# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 11:30:58 2025

@author: Luis1
"""

class mock_shutters:
    """Simula el backend para probar el frontend."""
    def __init__(self, n_shutters=4):
        # Dict con estados: "off", "on", "auto"
        self.shutter_states = {i: "off" for i in range(n_shutters)}