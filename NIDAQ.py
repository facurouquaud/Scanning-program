# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 09:59:14 2025

@author: Lenovo
https://github.com/ni/nidaqmx-python/blob/master/examples/synchronization/multi_function/ai_ao_sync.py
"""
from __future__ import annotations
import numpy as np
import threading
import dataclasses
import logging
from typing import Optional, List, Tuple
import nidaqmx
from nidaqmx.constants import AcquisitionType, Edge, Signal, TerminalConfiguration
# from nidaqmx.stream_writers import AnalogMultiChannelWriter
# from nidaqmx.errors import DaqError
import scan_parameters
import matplotlib.pyplot as plt
import time
import copy
from enum import Enum as _Enum
# from Shutters_backend import NIDAQShuttersBackend
from drivers.trajectories import generate_trajectory, scanning_2D  # , finish_scan
from base_scan import (BaseScan, ScanCallback, StartCallback, StopCallback,
                       ScanModeInfo)
import math
import sys

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


_DETECTORS = ["APD", "PMT"]
_MIN_X_UM, _MIN_Y_UM, _MIN_Z_UM = 0., 0., 0
_MAX_X_UM, _MAX_Y_UM, _MAX_Z_UM = 100., 100., 100.


class ScanModes(ScanModeInfo, _Enum):
    FORWARD = "Forward scan", 1, ["Imagen Ida",]
    BACKWARD = "Backward scan", 1, ["Imagen vuelta",]
    FULL_DATA = "Ida y vuelta", 2, ["Imagen Ida", "Imagen vuelta"]


_SCAN_MODES = {
    mode.description: mode
    for mode in ScanModes
    }


# Pixel filter generation functions

def make_pixel_filter_forward(n_pix: int, n_pix_acc: int):
    """Devuelve función de filtrado para imagen forward."""
    def forward_filter(line: np.ndarray):
        return np.asarray(line).reshape((1, 2, -1))[:, 0, n_pix_acc:n_pix_acc + n_pix]
    return forward_filter


def make_pixel_filter_backward(n_pix: int, n_pix_acc: int):
    """Devuelve función de filtrado para imagen forward."""
    def backward_filter(line: np.ndarray):
        return np.asarray(line).reshape((1, 2, -1))[:, 1, n_pix_acc + n_pix-1:n_pix_acc-1: -1]
    return backward_filter


def make_pixel_filter_full(n_pix: int, n_pix_acc: int):
    """Devuelve función de filtrado para imagen forward."""
    def full_filter(line: np.ndarray):
        line = np.asarray(line).reshape((2, -1,),)
        rv = np.empty((2, n_pix,), dtype=line.dtype)
        rv[0, :] = line[0, n_pix_acc:n_pix_acc + n_pix]
        rv[1, :] = line[1, n_pix_acc + n_pix:n_pix_acc:-1]
        return rv
    return full_filter


# Configuration dataclass
class _ScannerConfig:
    """NIDAQ specific Configuration."""

    def __init__(self,
                 device_name: str = "Dev1",
                 um_to_volts_DAQ: float = 0.04,
                 um_to_volts_NANO: float = 0.5,  # FIXME: revisar valor
                 ao_channels: List[str] = ["ao0", "ao1"],
                 ci_channel: str = "ctr0",
                 sample_rate: float = 100000.0,
                 max_voltage: float = 6,
                 scan_mode: ScanModes = ScanModes.FORWARD,
                 ):
        self.device_name = device_name
        self.um_to_volts_DAQ = um_to_volts_DAQ
        self.um_to_volts_NANO = um_to_volts_NANO
        self.ao_channels = ao_channels  # FIXME: nunca se usa
        self.ci_channel = ci_channel
        self.sample_rate = sample_rate
        self.max_voltage = max_voltage
        self.scan_mode = scan_mode


class _NIDAQScanThread(threading.Thread):
    """Thread for NIDAQ scanning operations."""

    def __init__(self, params, line_callbacks, volt_fast, volt_slow, fast_rel,
                 slow_rel, fast_back_v, slow_back_v,
                 samples_per_line: int,
                 total_samples: int, total_center_samples, total_back_samples,
                 line_indices: List[Tuple[int, int]],
                 true_px: int, n_px_acc: int, n_lines: int, acc:int, config: _ScannerConfig,
                 scan_end_callbacks: list[callable],
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scan_params = params
        self._line_callbacks = line_callbacks
        self._stop_callbacks = scan_end_callbacks
        self._stop_event = threading.Event()
        self._error_occurred = False
        self.n_px_acc = n_px_acc
        self.volt_fast = volt_fast
        self.volt_slow = volt_slow
        self.fast_rel = fast_rel
        self.slow_rel = slow_rel
        self.fast_back_v = fast_back_v
        self.slow_back_v = slow_back_v
        self.total_back_samples = total_back_samples
        self.true_px = true_px

        print(self.true_px)
        # Convert position to voltage (safe copy)

        # Validate voltage ranges
        # max_physical_DAQ = config.max_voltage / config.um_to_volts_DAQ
        # max_physical_NANO = config.max_voltage / config.um_to_volts_NANO
        # self.volt_x = np.clip(volt_x.copy(), -max_physical, max_physical) * config.um_to_volts
        # self.volt_y = np.clip(volt_y.copy(), -max_physical, max_physical) * config.um_to_volts
        # self.t = t

        self._validate_voltage(config.max_voltage)

        # Validación adicional

        if np.max(np.abs(self.volt_fast)) > config.max_voltage:
            actual_max = np.max(np.abs(self.volt_fast))
            raise ValueError(
                f"Voltaje X excede {config.max_voltage}V (llegó a {actual_max:.2f}V)\n"
                f"Revisar um_to_volts (actual: {config.um_to_volts_DAQ}) y tamaño de escaneo"
             )
        self.samples_per_line = samples_per_line
        self.n_lines = n_lines
        self.acc = acc
        self.config = config
        self.frames_samples = len(self.volt_fast)
        self.total_center_samples = total_center_samples
        # Pre-calculate line indices
        self.line_indices = [
            (i * samples_per_line, (i + 1) * samples_per_line)
            for i in range(n_lines)
        ]

    def _validate_voltage(self, max_voltage: float):
        """Ensure voltages are within DAQ limits with detailed logging."""
        x_min = np.min(self.volt_fast)
        x_max = np.max(self.volt_fast)
        y_min = np.min(self.volt_slow)
        y_max = np.max(self.volt_slow)

        if x_max > max_voltage or x_min < -max_voltage:
            logger.error(f"X voltage out of range: min={x_min:.2f}V, max={x_max:.2f}V, "
                         f"allowed=±{max_voltage}V")
            raise ValueError(f"X voltage out of range: {x_min:.2f}V to {x_max:.2f}V")

        if y_max > max_voltage or y_min < -max_voltage:
            logger.error(f"Y voltage out of range: min={y_min:.2f}V, max={y_max:.2f}V, "
                         f"allowed=±{max_voltage}V")
            raise ValueError(f"Y voltage out of range: {y_min:.2f}V to {y_max:.2f}V")

        logger.info(f"Voltage ranges validated: X({x_min:.2f}V to {x_max:.2f}V), "
                    f"Y({y_min:.2f}V to {y_max:.2f}V)")

    def pixel_filter(self, line: np.ndarray, n_pix: int, n_pix_acc: int):
        """
        Filtra píxeles válidos en una línea de escaneo.

        Args:
            line: Array de datos de la línea (1D)
            n_pix: Número de píxeles válidos
            n_pix_acc: Píxeles de aceleración

        Returns:
            Tuple: (pixeles_validos, pixeles_ida, pixeles_vuelta)
        """
        if not isinstance(line, np.ndarray):
            line = np.array(line)

        total_pixeles = len(line)
        puntos_por_linea = 2 * n_pix + 4 * n_pix_acc

        # Verificar longitud
        if total_pixeles != puntos_por_linea:
            raise ValueError(f"Longitud esperada: {puntos_por_linea}, recibida: {total_pixeles}")

        # Crear arrays de índices directamente
        idx_ida = np.arange(n_pix_acc, n_pix_acc + n_pix)
        inicio_vuelta = 3 * n_pix_acc + n_pix
        idx_vuelta = np.arange(inicio_vuelta, inicio_vuelta + n_pix)

        # Obtener los datos usando los índices
        line_ida = line[idx_ida]
        line_vuelta = line[idx_vuelta]
        line_valida = np.stack([line_ida, line_vuelta])

        return line_valida, line_ida, line_vuelta

    def channel_configuration(self, mode, ao_task, ci_task,
                              xy_signal, number_of_points, slow_chan, fast_chan):
        """Configura los canalaes"""
        ao_task.ao_channels.add_ao_voltage_chan(slow_chan)  # slow
        ao_task.ao_channels.add_ao_voltage_chan(fast_chan)  # fast

        # Export sample clock for synchronization
        ao_task.export_signals.export_signal(
            signal_id=nidaqmx.constants.Signal.SAMPLE_CLOCK,
            output_terminal="/Dev1/PFI0"
        )

        # Configure AO timing
        ao_task.timing.cfg_samp_clk_timing(
            rate=self.config.sample_rate,
            sample_mode=mode,
            samps_per_chan= number_of_points
        )

        # Configure counter input
        ci_chan = ci_task.ci_channels.add_ci_count_edges_chan(
            f"{self.config.device_name}/{self.config.ci_channel}",
            edge=Edge.RISING
        )
        ci_chan.ci_count_edges_count_reset_enable = True
        ci_chan.ci_count_edges_count_reset_term = ao_task.timing.samp_clk_term
        ci_chan.ci_count_edges_count_reset_active_edge = Edge.RISING

        # Sync CI with AO sample clock
        ci_task.timing.cfg_samp_clk_timing(
            rate=self.config.sample_rate,
            source=f"/{self.config.device_name}/ao/SampleClock",
            sample_mode=mode,
            samps_per_chan=number_of_points
        )
        # ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(ci_task.triggers.start_trigger.term)

        # writer = AnalogMultiChannelWriter(ao_task.out_stream, auto_start=False)
        number_of_samples_written_signal = ao_task.write(xy_signal, auto_start=False)

    def _execute_scan_stop_callbacks(self):
        """Notify all stop callbacks."""
        for callback in self._stop_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Stop callback error: {e}")

    def run(self):
        daq_acq_modes = [AcquisitionType.FINITE, AcquisitionType.CONTINUOUS]
        # FIXME:  ¿Esto no venía en config?
        chann_asign = {
            scan_parameters.RegionScanType.XY: {"fast": "Dev1/ao0", "slow": "Dev1/ao1"},
            scan_parameters.RegionScanType.XZ: {"fast": "Dev1/ao0", "slow": "Dev1/ao2"},
            scan_parameters.RegionScanType.YZ: {"fast": "Dev1/ao1", "slow": "Dev1/ao2"},
            }
        fast_chan = chann_asign[self.scan_params.scan_type]["fast"]
        slow_chan = chann_asign[self.scan_params.scan_type]["slow"]

        """Main scanning loop."""
        # Precompute center and flyback samples as integers
        flyback_samples = self.frames_samples - (self.n_lines * self.samples_per_line)

        # Ensure flyback_samples is non-negative
        flyback_samples = max(0, flyback_samples)

        # AO-only relocation - Mismo enfoque que el escaneo principal
        # FIXME: ¿Para que usamos CI acá?
        print("Yendo al origen")
        slow_0 = self.slow_rel[-1]
        fast_0 = self.fast_rel[-1]
        try:
            with nidaqmx.Task() as ai_task:  # vuelta a la posición de descanso
                # FIXME: BUILD FROM SCAN DATA (ojo Z)
                internal_read_channel_name_fast = "Dev1/_ao0_vs_aognd"
                internal_read_channel_name_slow = "Dev1/_ao1_vs_aognd"
                ai_task = nidaqmx.Task()
                ai_task.ai_channels.add_ai_voltage_chan(internal_read_channel_name_fast)
                ai_task.ai_channels.add_ai_voltage_chan(internal_read_channel_name_slow)
                last_position = ai_task.read()
            _, self.fast_init_frame, self.slow_init_frame, n_init = generate_trajectory(
                last_position, fast_0, slow_0, self.scan_params.dwell_time,
                a_max_fast=self.acc, a_max_slow=self.acc
            )
            xy_init_signal = np.vstack((self.slow_init_frame , self.fast_init_frame))
            n_init_samples = len(self.fast_init_frame)
            print(f"la ida tiene {n_init_samples}")

            with nidaqmx.Task() as ao_task, nidaqmx.Task() as ci_task:
                self.channel_configuration(
                    daq_acq_modes[0], ao_task, ci_task, xy_init_signal,
                    n_init_samples, slow_chan, fast_chan
                    )
                # Start tasks - CI first then AO
                ci_task.start()
                ao_task.start()
                # ao_task.write(xy_signal, auto_start=True)
                ci_task.read(
                    number_of_samples_per_channel=n_init_samples)
            logger.info("Relocación al cero completada correctamente.")
        except Exception as e:
            logger.error(f"Error en relocación: {e}")
            self._stop_event.set()
            return

        print("Frames")
        try:
            frame_count = 0
            with nidaqmx.Task() as ao_task, nidaqmx.Task() as ci_task:
                xy_signal = np.vstack((self.volt_slow, self.volt_fast))
                # Configure analog
                self.channel_configuration(
                    daq_acq_modes[1], ao_task, ci_task, xy_signal,
                    self.frames_samples + flyback_samples, slow_chan, fast_chan
                    )
                # Arrancar los tasks de esta forma no debe sincronizarlos
                # Hacer algo parecido a esto
                # https://github.com/ni/nidaqmx-python/issues/162
            
                ci_task.start()
                ao_task.start()
                while not self._stop_event.is_set():
                    # Process line by line
                    logger.info("Arrancando scan")
                    for line_idx, (start, end) in enumerate(self.line_indices):
                        # Read line data
                        try:
                            line_total_data = ci_task.read(
                                number_of_samples_per_channel=self.samples_per_line
                            )
                        except Exception as e:  # nidaqmx.errors.DaqError
                            logger.error(f"DAQ read error on line {line_idx}: {e}")
                            self._stop_event.set()
                            break
                        # print(sum(line_total_data))
                        try:
                            # Process data
                            line_data_both, line_data_first, line_data_second = self.pixel_filter(
                                line_total_data,
                                self.true_px,
                                self.n_px_acc
                            )

                            # Select data based on scan mode
                            if self.config.scan_mode is ScanModes.FORWARD:
                                current_line = line_data_first
                            elif self.config.scan_mode is ScanModes.BACKWARD:
                                current_line = line_data_second
                            else:  # ScanModes.FULL_DATA
                                current_line = line_data_both

                            last_line = (line_idx == self.n_lines - 1)

                            # Send to callbacks
                            for callback in self._line_callbacks:
                                try:
                                    if callback(current_line, line_idx, last_line):
                                        self._stop_event.set()
                                except Exception as e:
                                    logging.error(f"Callback error on line {line_idx}: {e}")

                            # if self._stop_event.is_set():
                            #       logger.info("Scan stopped by user request")
                            #       break

                        except Exception as e:
                            logger.error(f"Processing error on line {line_idx}: {e}")
                            self._stop_event.set()
                            self._error_occurred  = True
                            break
                          # Darle tiempo a otros threads
                    # Read and discard flyback samples at end of frame
                    print("Volviendo al origen")
                    if flyback_samples > 0:
                        try:
                            ci_task.read(
                                number_of_samples_per_channel=flyback_samples
                            )
                        except Exception as e:
                            logger.warning(f"Flyback read skipped: {e}")
                            self._error_occurred = True
                # Aca termino un frame y su flyback
                    frame_count += 1
                    logger.info(f"Completed frame {frame_count}")
                # End of NI-DAQ task
            # End of frame processing
        except Exception as e:
            logger.error(f"Critical scan error: {e}", exc_info=True)
            self._error_occurred = True
            self._stop_event.set()
        try:
                with nidaqmx.Task() as ai_task:
                    internal_read_channel_name_fast = "Dev1/_ao0_vs_aognd"
                    internal_read_channel_name_slow  = "Dev1/_ao1_vs_aognd"
                    ai_task = nidaqmx.Task()
                    ai_task.ai_channels.add_ai_voltage_chan(internal_read_channel_name_fast)
                    ai_task.ai_channels.add_ai_voltage_chan(internal_read_channel_name_slow)
                    last_position = ai_task.read()        
                         
                _, self.fast_back_v, self.slow_back_v, n_rel  = generate_trajectory(
            last_position, 0, 0, self.scan_params.dwell_time, a_max_fast=self.acc, a_max_slow=self.acc
                )
                xy_back_signal = np.vstack((self.slow_back_v, self.fast_back_v))
                n_reloc_samples = len(self.fast_back_v)
                print(f"la vuelta tiene {n_reloc_samples}")
                with nidaqmx.Task() as ao_task, nidaqmx.Task() as ci_task:
                    self.channel_configuration(
                        daq_acq_modes[0], ao_task, ci_task, xy_back_signal,
                        n_reloc_samples, slow_chan, fast_chan
                        )
                    # Start tasks - CI first then AO
                    ci_task.start()
                    ao_task.start()
                    # ao_task.write(xy_signal, auto_start=True)
                    ci_task.read(
                        number_of_samples_per_channel=n_reloc_samples)
                logger.info("Relocación al cero completada correctamente.")
        except Exception as e:
                    logger.error(f"Error en relocación: {e}")
                    self._stop_event.set()
                    # return
        logger.info("Scan completed successfully")
        self._execute_scan_stop_callbacks()


    def stop(self):
        """Gracefully stop the scan."""
        self._stop_event.set()
        logger.info("Pidiéndole al thread que pare")


class NIDAQScan(BaseScan):
    """NIDAQ-based scanner for microscopy systems."""

    def __init__(self, config: Optional[_ScannerConfig] = None):
        self.config = config or _ScannerConfig()
        self._start_callbacks: List[StartCallback] = []
        self._stop_callbacks: List[StopCallback] = []
        self._line_callbacks: List[ScanCallback] = []
        self._thread: Optional[_NIDAQScanThread] = None
        self.scan_params: Optional[scan_parameters.RegionScanParameters] = None
        self._scanning = False
        self.current_position = np.array([0.0, 0.0])

    def register_callbacks(self,
                           scan_start_callback: Optional[StartCallback] = None,
                           scan_end_callback: Optional[StopCallback] = None,
                           line_callback: Optional[ScanCallback] = None):
        """Register scan callbacks."""
        if scan_start_callback:
            self._start_callbacks.append(scan_start_callback)
        if scan_end_callback:
            self._stop_callbacks.append(scan_end_callback)
        if line_callback:
            self._line_callbacks.append(line_callback)

    @property
    def sfast(self) -> float:
        """Número de píxeles en la dirección rápida."""
        if not self.scan_params:
            raise AttributeError("scan_params not set")
        return self.scan_params.line_length_fast

    @property
    def sslow(self) -> float:
        if not self.scan_params:
            raise AttributeError("scan_params not set")
        return self.scan_params.line_length_slow

    def start_scan(self,
                   params: scan_parameters.RegionScanParameters,
                   scan_mode_name: str,
                   ):
        """Start a new scan with given parameters."""
        if self._scanning:
            logger.warning(
                "Nuevo scan iniciado mientras el anterior estaba en marcha"
                )
            return False
        scan_mode = _SCAN_MODES.get(scan_mode_name, None)
        if not scan_mode:
            logger.warning(
                "Modo de escaneo no reconocido: %s", scan_mode_name
                )
            return False
        self.scan_params = dataclasses.replace(params)
        self.scan_mode = scan_mode
        if not self._validate_scan_params(params):
            raise ValueError("Invalid scan parameters")

        self._scanning = True

        self._execute_scan_start_callbacks()
        self._thread = self._create_scan_thread()
        self._thread.start()

    def _validate_scan_params(self, params: scan_parameters.RegionScanParameters) -> bool:
        valid = True
        if params.line_length_fast <= 0 or params.line_length_slow <= 0:
            logger.error("Scan size must be positive")
            valid = False
        if params.dwell_time <= 1E-1:
            logger.error("Dwell time too small")
            valid = False
        if params.pixel_size <= 0:
            logger.error("Pixel size must be positive")
            valid = False
        # FIXME: Esto asume XY, podría ser Z
        # if (_MIN_X_UM < params.line_length_fast < _MAX_X_UM or
        #         _MIN_Y_UM < params.line_length_slow < _MAX_Y_UM):
        #     logger.error("Region out of range")
        #     valid = False
        return valid

    def _execute_scan_start_callbacks(self):
        """Notify all start callbacks."""
        data_shape = self.get_data_shape()
        for callback in self._start_callbacks:
            try:
                callback(self.scan_params, data_shape, self.scan_mode.images_per_frame)
            except Exception as e:
                logger.error("Start callback error %s: %s", type(e), e)

    def cleanup_scan(self):
        if self._thread:
            if self._thread.is_alive():
                logger.error("Can not cleanup while scanning")
            else:
                self._thread.join()
                self._thread = None

    def stop_scan(self):
        """Stop the current scan."""
        if not self._scanning:
            logger.warning("No active scan to stop")
            return

        self._scanning = False

        # Then stop the thread
        if not self._thread:
            logger.error("Can not stop: no thread found")
            return
        self._thread.stop()

        # Wait with reasonable timeout (e.g., 2 seconds)
        self._thread.join(timeout=100)
        # if self._thread.is_alive():
        #     logger.error("Failed to stop scan thread within timeout")
        #     # Force cleanup if needed
        #     self._thread = None
        # else:
        #     self._thread = None
        #     self._execute_scan_stop_callbacks()

    def _calculate_parameters(self, scan_width_fast: float, scan_width_slow: float, pixel_size: float,
                          dwell_time: float, n_pix: int) -> Tuple:
        """
        Calcula parámetros de escaneo.

        REQUISITOS
        # Tiempo mínimo de aceleración para alcanzar v_f con a_max_si
        t_acc_min = v_f / a_max_si  # s
     DE UNIDADES:
          - pixel_size: metros (m)
          - dwell_time: segundos (s)
          - scan_width_x/scan_width_slow: metros (m)
        Devuelve:
          x0 (m), n_pix, n_acc (muestras), n_acc_min, v_f (m/s), acc (m/s^2), v_f (repetido)
        """
        self.a_max = 130  # µm/ms²

        # --- PARÁMETROS FIJOS (documentados en SI) ---
        # Originalmente 130 µm/ms^2 -> equivale a 130 m/s^2 (ver conversión en la explicación)

        # Validaciones básicas
        if pixel_size <= 0 or dwell_time <= 0:
            self._execute_scan_stop_callbacks()
            raise ValueError("Parameters must be positive")

        # Debug: imprimir entradas (m, s)
        print("DEBUG _calculate_parameters inputs:", file=sys.stderr)
        print(f"  scan_width_fast (m): {scan_width_fast}", file=sys.stderr)
        print(f"  scan_width_slow (m): {scan_width_slow}", file=sys.stderr)
        print(f"  pixel_size (m): {pixel_size}", file=sys.stderr)
        print(f"  dwell_time (s): {dwell_time}", file=sys.stderr)
        print(f"  n_pix: {n_pix}", file=sys.stderr)

        # v_f en m/s (velocidad de muestreo = tamaño de píxel / tiempo por píxel)
        v_f = pixel_size / dwell_time
        t_acc_min = v_f / self.a_max
        n_acc_min = math.ceil(t_acc_min / dwell_time)
        n_acc = max(int(n_acc_min), 4)  # Asegurar mínimo 4
        acc = v_f / (n_acc * dwell_time)

        # Debug: imprimir intermedios
        print("DEBUG _calculate_parameters internals:", file=sys.stderr)
        print(f"  v_f (m/s): {v_f}", file=sys.stderr)
        print(f"  t_acc_min (s): {t_acc_min}", file=sys.stderr)
        print(f"  n_acc_min (samples): {n_acc_min}", file=sys.stderr)
        print(f"  n_acc (samples, used): {n_acc}", file=sys.stderr)
        print(f"  acc (m/s^2): {acc}", file=sys.stderr)

        # Comprobaciones
        if scan_width_fast > 100 or scan_width_slow > 100:  # si estas en metros, 100 m es enorme; tal vez querías µm
            self._execute_scan_stop_callbacks()
            raise ValueError("Region out of range")

        if acc > self.a_max * 1.0001:  # margen pequeño
            self._execute_scan_stop_callbacks()
            raise ValueError("Acceleration out of range")

        if n_pix > 500:
            self._execute_scan_stop_callbacks()
            raise ValueError("number of pixels out of range")

        # x0: distancia recorrida durante la aceleración (en metros)
        fast0 = 0.5 * acc * ((n_acc * dwell_time) ** 2)

        return fast0, n_pix, n_acc, n_acc_min, v_f, acc, v_f

    def update_current_position(self, fast: float, slow: float):
        """Actualiza la posición actual del escáner."""
        self.current_position = np.array([fast, slow])
        logger.info(f"Posición actual actualizada: ({fast}, {slow}) µm")

    def _create_scan_thread(self) -> _NIDAQScanThread:

        # Convertir todo a metros consistentemente
        params = self.scan_params
        dwell_time = float(params.dwell_time * 1E-6)  # µs -> s
        px_size = float(params.pixel_size * 1E-6)     # µm -> m
        sfast = float(params.line_length_fast * 1E-6)       # µm -> m
        sslow = float(params.line_length_slow * 1E-6)       # µm -> m
        # start_slow = float(params.start_point[1] * 1E-6)  # µm -> m
        n_pix = int(params.true_px_num)

        required_sample_rate = 1.0 / (dwell_time)

        # Crear configuración temporal para el thread
        thread_config = _ScannerConfig(
            device_name=self.config.device_name,
            um_to_volts_DAQ=self.config.um_to_volts_DAQ,
            um_to_volts_NANO=self.config.um_to_volts_NANO,
            ao_channels=self.config.ao_channels,
            ci_channel=self.config.ci_channel,
            sample_rate=required_sample_rate,  # Usar el nuevo sample rate
            max_voltage=self.config.max_voltage,
            scan_mode=self.scan_mode,
        )

        chann_asign = {
                    "XY": {"fast": self.config.um_to_volts_DAQ, "slow": self.config.um_to_volts_DAQ},
                    "XZ": {"fast": self.config.um_to_volts_DAQ, "slow": self.config.um_to_volts_NANO},
                    "YZ": {"fast": self.config.um_to_volts_DAQ, "slow": self.config.um_to_volts_NANO}
                    }

        fast_chan_convertion = chann_asign[self.scan_params.scan_type.value]["fast"]
        slow_chan_convertion = chann_asign[self.scan_params.scan_type.value]["slow"]
        
        channel_offset = {
                    "XY": {"fast": 0, "slow": 0},
                    "XZ": {"fast": 0, "slow": 4.5},
                    "YZ": {"fast": 0, "slow": 4.5}
                    }
        slow_chan_offset = channel_offset[self.scan_params.scan_type.value]["slow"]

        # Calcular parámetros
        fast0, true_px, n_px_acc, _, v_f, acc, _ = self._calculate_parameters(
            sfast, sslow, px_size, dwell_time, n_pix
        )

        # generar recentrado solo si haslow center en params
        # x_0, y_0 = self.current_position
        center = getattr(params, "center", None)

        if center is not None:
            # Preferir prev_center si existe (centro anterior en µm)
            fast_f_um = copy.deepcopy(params.start_point[0] * 1E-6)
            slow_f_um = copy.deepcopy(2*params.end_point[1] * 1E-6)
            # x_f_um = params.start_point[0]*2
            # y_f_um =  params.start_point[1]*2

            t_rel, fast_rel_um, slow_rel_um, n_rel = generate_trajectory(
                self.current_position,
                fast_f_um,
                slow_f_um,
                dwell_time,
                a_max_fast=acc,
                a_max_slow=acc
                )
                # convertir a V (si config.um_to_volts es V/µm)

            fast_rel_v = fast_rel_um * fast_chan_convertion
            slow_rel_v = slow_rel_um * slow_chan_convertion

            # asegurar al menos 2 puntos
            if fast_rel_v.size < 2:
                fast_rel_v = np.pad(fast_rel_v, (0, 2 - fast_rel_v.size), mode='edge')
            if slow_rel_v.size < 2:
                slow_rel_v = np.pad(slow_rel_v, (0, 2 - slow_rel_v.size), mode='edge')

            # clip por seguridad
            fast_rel_v = np.clip(fast_rel_v, -self.config.max_voltage, self.config.max_voltage)
            slow_rel_v = np.clip(slow_rel_v, -self.config.max_voltage, self.config.max_voltage)
            self.update_current_position(fast_f_um, slow_f_um)
        if sslow > sfast:
            n_lines = int(sslow / (px_size))
        else:
            n_lines = int(sfast / px_size)

        # --- Asumo: x_rel_um, y_rel_um están en metros (rename para claridad) ---
        fast_rel_m = fast_rel_um  # si realmente son metros
        slow_rel_m = slow_rel_um

        # Asegurar >=2 puntos (en metros)
        if fast_rel_m.size < 2:
            fast_rel_m = np.pad(fast_rel_m, (0, 2 - fast_rel_m.size), mode='edge')
        if slow_rel_m.size < 2:
            slow_rel_m = np.pad(slow_rel_m, (0, 2 - slow_rel_m.size), mode='edge')
        logger.info("La cantidad de pixeles por linea es %s", true_px)
        # Trajectoria de escaneo: escaneo2D_back devuelve arrays en metros (supuesto)
        t, scan_fast_m, scan_slow_m, samples_per_line = scanning_2D(
            n_lines, fast0, params.end_point[1] * 1E-6, dwell_time, n_px_acc, true_px, acc, v_f, px_size
        )

        # --- CALCULAR OFFSETS EN METROS para asegurar continuidad ---
        # último punto del recentrado (en metros)
        last_center_fast_m = fast_rel_m[-1]
        last_center_slow_m = slow_rel_m[-1]

        # offset para que el primer punto del escaneo coincida con final del recentrado
        offset_scan_fast_m = last_center_fast_m - scan_fast_m[0]
        offset_scan_slow_m = last_center_slow_m - scan_slow_m[0]

        # aplicar offset (en metros)
        scan_fast_m_shifted = scan_fast_m + offset_scan_fast_m
        scan_slow_m_shifted = scan_slow_m + offset_scan_slow_m

        # ahora generar vuelta al origen (back) en metros
        t_back, back_fast_m, back_slow_m, n_back = generate_trajectory(
            self.current_position, 0, 0, dwell_time, a_max_fast=acc, a_max_slow=acc
        )
        self.update_current_position(0, 0)
        # offset para que el primer punto de la vuelta coincida con el último punto del escaneo
        last_scan_fast_m = scan_fast_m_shifted[-1]
        last_scan_slow_m = scan_slow_m_shifted[-1]
        offset_back_fast_m = last_scan_fast_m - back_fast_m[0]
        offset_back_slow_m = last_scan_slow_m - back_slow_m[0]

        back_fast_m_shifted = back_fast_m + offset_back_fast_m
        back_slow_m_shifted = back_slow_m + offset_back_slow_m

        # --- Convertir todo a voltios UNA VEZ (V/µm * 1e6 µm/m) ---

        volt = 1E6
        fast_rel_v = fast_rel_m * fast_chan_convertion * volt
        slow_rel_v = slow_rel_m * slow_chan_convertion * volt 

        volt_fast = scan_fast_m_shifted * fast_chan_convertion*volt
        volt_slow = scan_slow_m_shifted * slow_chan_convertion*volt 

        fast_back_v = back_fast_m_shifted * fast_chan_convertion*volt
        slow_back_v = back_slow_m_shifted * slow_chan_convertion*volt 

        # Clipping (chequeamos que no superen valores de voltajes)
        volt_fast = np.clip(volt_fast, -self.config.max_voltage, self.config.max_voltage)
        volt_slow = np.clip(volt_slow, -self.config.max_voltage, self.config.max_voltage)
        fast_rel_v = np.clip(fast_rel_v, -self.config.max_voltage, self.config.max_voltage)
        slow_rel_v = np.clip(slow_rel_v, -self.config.max_voltage, self.config.max_voltage)
        fast_back_v = np.clip(fast_back_v, -self.config.max_voltage, self.config.max_voltage)
        slow_back_v = np.clip(slow_back_v, -self.config.max_voltage, self.config.max_voltage)

        # --- Comprobaciones rápidas ---
        print("Continuidad center->scan (V):", fast_rel_v[-1], volt_fast[0], "diff:", fast_rel_v[-1]-volt_fast[0])
        print("Continuidad scan->back (V):", volt_fast[-1], fast_back_v[0], "diff:", volt_fast[-1]-fast_back_v[0])
        print(len(volt_fast) + len(fast_rel_v) + len(fast_back_v))
        # Asserts (opcional, lanzar error si no coincide dentro de tol)
        tol = 1e-6
        assert np.allclose(fast_rel_v[-1], volt_fast[0], atol=tol), "No coincide fast center->scan"
        assert np.allclose(slow_rel_v[-1], volt_slow[0], atol=tol), "No coincide slow center->scan"
        assert np.allclose(volt_fast[-1], fast_back_v[0], atol=tol), "No coincide fast scan->back"
        assert np.allclose(volt_slow[-1], slow_back_v[0], atol=tol), "No coincide slow scan->back"
        total_samples = len(volt_fast)
        total_center_samples = len(fast_rel_v)
        total_back_samples = len(fast_back_v)
        #  # construir line_indices igual que antes
        line_indices = [(i * samples_per_line, (i + 1) * samples_per_line) for i in range(n_lines)]

        # Esta parte grafica los voltajes generados. Mantener comentado a la hora de escanear.
        def muestra_escaneo(titulo,t,x,y): #grafica lo que le mandamos a los espejos en voltaje
            fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6), sharex=True)
            ax1.scatter(t, x,s = 8, color ="black")
            ax1.set_ylabel("Posición X [V]")
            ax1.set_title(titulo)
            ax1.grid(True)
            ax2.scatter(t, y,s = 8, color="black")
            ax2.set_ylabel("Posición z [V]")
            ax2.set_xlabel("Tiempo [us]")
            ax2.grid(True)
            plt.tight_layout()
            plt.show()
        if len(t_rel) > 1:
              muestra_escaneo("Voltajes de recentrado",t_rel,fast_rel_v,slow_rel_v)
        muestra_escaneo(f"Escaneo de {n_lines + 1} de escaneo", t, volt_fast, volt_slow)
        if len(t_back) > 1:
                muestra_escaneo("Voltajes de vuelta al origen",t_back,fast_back_v,slow_back_v)

        # devolver/crear thread pasando solo señales slowa procesadas:
        return _NIDAQScanThread(
            params=params,
            line_callbacks=self._line_callbacks,
            volt_fast=volt_fast,
            volt_slow=volt_slow,
            fast_rel=fast_rel_v,
            slow_rel=slow_rel_v,
            fast_back_v=fast_back_v,
            slow_back_v=slow_back_v,
            samples_per_line=samples_per_line,
            total_samples=total_samples,
            line_indices=line_indices,
            total_center_samples=total_center_samples,
            total_back_samples=total_back_samples,
            true_px=true_px,
            n_px_acc=n_px_acc,
            n_lines=n_lines,
            acc = acc,
            config=thread_config
        )
        print(f"{n_px_acc=}")

    def get_data_shape(self) -> Tuple[int, int]:
        """Get scan data dimensions."""
        if not self.scan_params:
            return (0, 0)
        return (int(self.sfast / self.scan_params.pixel_size),
                int(self.sslow / self.scan_params.pixel_size))

    def get_extents(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Get scan extents in physical units."""
        # FIXME: Añadir Z también
        if not self.scan_params:
            return ((_MIN_X_UM, _MIN_Y_UM), (_MAX_X_UM, _MAX_Y_UM),)

    def get_detectors(self) -> list[str]:
        return _DETECTORS

    def get_scan_modes(self) -> list[ScanModeInfo]:
        return [modo.value for modo in ScanModes]
