# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 09:59:14 2025

@author: Lenovo
"""
import numpy as np
import math
import threading
import dataclasses
import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional, List, Tuple
import nidaqmx
from nidaqmx.constants import AcquisitionType, Edge, Signal, TerminalConfiguration
from nidaqmx.stream_writers import AnalogMultiChannelWriter
from nidaqmx.errors import DaqError
import scan_parameters
import matplotlib.pyplot as plt
from scan_parameters import RegionScanData
from typing import Tuple
import time
import copy
from Shutters_backend import NIDAQShuttersBackend

# Define callback types
StartCallback = Callable[[object, Tuple[int, int]], None]
StopCallback = Callable[[], None]
ScanCallback = Callable[[np.ndarray, int, bool], None]

# Configuration dataclass
class ScannerConfig:
    def __init__(self,
                 device_name: str = "Dev1",
                 um_to_volts_DAQ: float = 0.04,
                 um_to_volts_NANO: float = 0.001, #VER BIEN 
                 ao_channels: List[str] = ["ao0", "ao1"],
                 ci_channel: str = "ctr0",
                 sample_rate: float = 100000.0,
                 max_voltage: float = 5):
        self.device_name = device_name
        self.um_to_volts_DAQ = um_to_volts_DAQ
        self.um_to_volts_NANO = um_to_volts_NANO 
        self.ao_channels = ao_channels
        self.ci_channel = ci_channel
        self.sample_rate = sample_rate
        self.max_voltage = max_voltage
        self._scanning = False


class BaseScan(ABC):
    """Abstract base class for scanners."""
    
    @abstractmethod
    def register_callbacks(self,
                          scan_start_callback: Optional[StartCallback],
                          scan_end_callback: Optional[StopCallback],
                          line_callback: Optional[ScanCallback]):
        pass

    @abstractmethod
    def get_data_shape(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def start_scan(self, params: scan_parameters.RegionScanParameters):
        pass

    @abstractmethod
    def stop_scan(self):
        pass

    @abstractmethod
    def get_extents(self) -> Tuple[Tuple[float, float], 
                                  Tuple[float, float]]:
        pass


class _NIDAQScanThread(threading.Thread):
    """Thread for NIDAQ scanning operations."""
    
    def __init__(self, params,line_callbacks,volt_fast, volt_slow,fast_rel, slow_rel,fast_back_v, slow_back_v,
                 samples_per_line: int,
                 total_samples: int,total_center_samples,total_back_samples, line_indices: List[Tuple[int,int]],
                 true_px: int, n_px_acc: int, n_lines: int, config: ScannerConfig,
                 *args, **kwargs):

                  
        super().__init__(*args, **kwargs)
        self.scan_params = params
        self._line_callbacks = line_callbacks
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
        max_physical_DAQ = config.max_voltage / config.um_to_volts_DAQ
        max_physical_NANO = config.max_voltage / config.um_to_volts_NANO
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
        self.config = config
        self.frames_samples = len(self.volt_fast)
        self.total_center_samples =  total_center_samples
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
            logging.error(f"X voltage out of range: min={x_min:.2f}V, max={x_max:.2f}V, "
                          f"allowed=±{max_voltage}V")
            raise ValueError(f"X voltage out of range: {x_min:.2f}V to {x_max:.2f}V")
       
        if y_max > max_voltage or y_min < -max_voltage:
            logging.error(f"Y voltage out of range: min={y_min:.2f}V, max={y_max:.2f}V, "
                          f"allowed=±{max_voltage}V")
            raise ValueError(f"Y voltage out of range: {y_min:.2f}V to {y_max:.2f}V")
       
        logging.info(f"Voltage ranges validated: X({x_min:.2f}V to {x_max:.2f}V), "
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
        line_valida = np.concatenate([line_ida, line_vuelta])
        
        return line_valida, line_ida, line_vuelta
    


    def run(self):
        chann_asign = {
                    "XY": {"fast": "Dev1/ao0", "slow": "Dev1/ao1"},
                    "XZ": {"fast": "Dev1/ao0", "slow": "Dev1/ao2"},
                    "YZ": {"fast": "Dev1/ao1", "slow": "Dev1/ao2"},
                }   
        fast_chan = chann_asign[self.scan_params.scan_type.value]["fast"]
        slow_chan = chann_asign[self.scan_params.scan_type.value]["slow"]

        """Main scanning loop."""
        last_line_last_pixel = None
        processed_data = []
        
        # Precompute center and flyback samples as integers
        center_samples = int(round(self.total_center_samples))
        flyback_samples = self.frames_samples - (self.n_lines * self.samples_per_line) 
        
        # Ensure flyback_samples is non-negative
        flyback_samples = max(0, flyback_samples)
        
        # AO-only relocation - Mismo enfoque que el escaneo principal
        if self.fast_rel is not None and self.slow_rel is not None:
            try:
                xy_reloc_signal = np.vstack((self.slow_rel, self.fast_rel))
                n_reloc_samples = len(self.fast_rel)
                
                with nidaqmx.Task() as ao_task, nidaqmx.Task() as ci_task:
                    # Configurar canales AO igual que en el escaneo principal
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
                        sample_mode=AcquisitionType.FINITE,
                        samps_per_chan= n_reloc_samples
                    )
                    
                    # Configure counter input
                    ci_chan = ci_task.ci_channels.add_ci_count_edges_chan(
                        f"{self.config.device_name}/{self.config.ci_channel}",
                        edge=Edge.RISING
                    )
                    ci_chan.ci_count_edges_count_reset_enable = True
                    ci_chan.ci_count_edges_count_reset_term =  ao_task.timing.samp_clk_term
                    ci_chan.ci_count_edges_count_reset_active_edge = Edge.RISING
                    
                    # Sync CI with AO sample clock
                    ci_task.timing.cfg_samp_clk_timing(
                        rate=self.config.sample_rate,
                        source=f"/{self.config.device_name}/ao/SampleClock",
                        sample_mode=AcquisitionType.FINITE,
                        samps_per_chan = n_reloc_samples
                    )
                    
                    writer = AnalogMultiChannelWriter(ao_task.out_stream, auto_start=False)
                    number_of_samples_written_signal = ao_task.write(xy_reloc_signal, auto_start=False)
                    
                    # Start tasks - CI first then AO
                    ci_task.start()
                    ao_task.start()
                    ci_task.read(
                        number_of_samples_per_channel=n_reloc_samples,
                        timeout=2.0
                    )
                    
                   
                logging.info("Relocación completada correctamente.")
            except Exception as e:
                logging.error(f"Error en relocación: {e}")
                self._stop_event.set()
                return
    
        try:
            frame_count = 0
            self._scanning = True
            while not self._stop_event.is_set() and self._scanning:
                xy_signal = np.vstack((self.volt_slow, self.volt_fast))
                
                with nidaqmx.Task() as ao_task, nidaqmx.Task() as ci_task:
                    # Configure analog output
                    ao_task.ao_channels.add_ao_voltage_chan(slow_chan, name_to_assign_to_channel="slow")
                    ao_task.ao_channels.add_ao_voltage_chan(fast_chan, name_to_assign_to_channel="fast")

                    
                    # Export sample clock for synchronization
                    ao_task.export_signals.export_signal(
                        signal_id=nidaqmx.constants.Signal.SAMPLE_CLOCK,
                        output_terminal="/Dev1/PFI0"
                    )
                    
                    # Configure AO timing
                    ao_task.timing.cfg_samp_clk_timing(
                        rate=self.config.sample_rate,
                        sample_mode=AcquisitionType.FINITE,
                        samps_per_chan= self.frames_samples + flyback_samples
                    )
                    
                    # Configure counter input
                    ci_chan = ci_task.ci_channels.add_ci_count_edges_chan(
                        f"{self.config.device_name}/{self.config.ci_channel}",
                        edge=Edge.RISING
                    )
                    ci_chan.ci_count_edges_count_reset_enable = True
                    ci_chan.ci_count_edges_count_reset_term =  ao_task.timing.samp_clk_term
                    ci_chan.ci_count_edges_count_reset_active_edge = Edge.RISING
                    
                    # Sync CI with AO sample clock
                    ci_task.timing.cfg_samp_clk_timing(
                        rate=self.config.sample_rate,
                        source=f"/{self.config.device_name}/ao/SampleClock",
                        sample_mode=AcquisitionType.FINITE,
                        samps_per_chan = self.frames_samples + flyback_samples
                    )
                    
                    writer = AnalogMultiChannelWriter(ao_task.out_stream, auto_start=False)
                    number_of_samples_written_signal = ao_task.write(xy_signal, auto_start=False)
                    
                    # Start tasks - CI first then AO
                    ci_task.start()
                    ao_task.start()
                    # ao_task.write(xy_signal, auto_start=True)
                    
                    # Process line by line
                    for line_idx, (start, end) in enumerate(self.line_indices):
                        if self._stop_event.is_set():
                            logging.info("Scan stopped by user request")
                            self._scan_completed = False
                            break
                        
                        # Read line data
                        try:
                            line_total_data = ci_task.read(
                                number_of_samples_per_channel=self.samples_per_line,
                                timeout=4.0
                            )
                        except Exception as e:
                            logging.error(f"DAQ read error on line {line_idx}: {e}")
                            self._stop_event.set()
                            break
                        
                        # Validate data length
                        if len(line_total_data) != self.samples_per_line:
                            logging.warning(
                                f"Line {line_idx} length mismatch: "
                                f"expected {self.samples_per_line}, got {len(line_total_data)}"
                            )
                            continue
                        
                        try:
                            # Process data
                            line_data_both, line_data_first, line_data_second = self.pixel_filter(
                                line_total_data,
                                self.true_px,
                                self.n_px_acc
                            )
                            
                            # Select data based on scan mode
                            if self.scan_params.scan_data.name == 'FIRST':
                                current_line = line_data_first
                            elif self.scan_params.scan_data.name == 'SECOND':
                                current_line = line_data_second
                            else:  # 'BOTH'
                                current_line = line_data_both
                            
                            # # Real-time processing
                            # normalized_line = current_line if last_line_last_pixel is None \
                            #     else current_line - last_line_last_pixel
                            
                            # last_line_last_pixel = current_line[-1]  # Update for next line
                            
                            # diff_line = np.insert(np.diff(normalized_line), 0, 0)
                            # processed_line = diff_line
                            
                            # # Store for visualization
                            # processed_data.append(processed_line)
                            
                            last_line = (line_idx == self.n_lines - 1)
                            
                            # Send to callbacks
                            for callback in self._line_callbacks:
                                try:
                                    if callback(line_data_first, line_idx, last_line):
                                        self._stop_event.set()
                                except Exception as e:
                                    logging.error(f"Callback error on line {line_idx}: {e}")
                        
                        except Exception as e:
                            logging.error(f"Processing error on line {line_idx}: {e}")
                            self._stop_event.set()
                            break
                    
                    # Read and discard flyback samples at end of frame
                    if flyback_samples > 0 and not self._stop_event.is_set():
                        try:
                            ci_task.read(
                                number_of_samples_per_channel=flyback_samples,
                                timeout=2.0
                            )
                        except Exception as e:
                            logging.warning(f"Flyback read skipped: {e}")
                
                # End of frame processing
                frame_count += 1
                logging.info(f"Completed frame {frame_count}")
                last_line_last_pixel = None  # Reset for next frame

        
        except Exception as e:
            logging.error(f"Critical scan error: {e}", exc_info=True)
            self._error_occurred = True
            self._scanning = False
        
        finally:
            self._stop_event.set()
            if self._stop_event.is_set():
                try:
                    xy_back_signal = np.vstack((self.slow_back_v, self.fast_back_v))
                    n_reloc_samples = len(self.fast_back_v)

                    with nidaqmx.Task() as ao_task, nidaqmx.Task() as ci_task:
                        # Configurar canales AO igual que en el escaneo principal
                        ao_task.ao_channels.add_ao_voltage_chan(slow_chan)  # Y
                        ao_task.ao_channels.add_ao_voltage_chan(fast_chan)  # X
                        # Export sample clock for synchronization
                        ao_task.export_signals.export_signal(
                            signal_id=nidaqmx.constants.Signal.SAMPLE_CLOCK,
                            output_terminal="/Dev1/PFI0"
                        )
                        
                        # Configure AO timing
                        ao_task.timing.cfg_samp_clk_timing(
                            rate=self.config.sample_rate,
                            sample_mode=AcquisitionType.FINITE,
                            samps_per_chan= n_reloc_samples
                        )
                        
                        # Configure counter input
                        ci_chan = ci_task.ci_channels.add_ci_count_edges_chan(
                            f"{self.config.device_name}/{self.config.ci_channel}",
                            edge=Edge.RISING
                        )
                        ci_chan.ci_count_edges_count_reset_enable = True
                        ci_chan.ci_count_edges_count_reset_term =  ao_task.timing.samp_clk_term
                        ci_chan.ci_count_edges_count_reset_active_edge = Edge.RISING
                        
                        # Sync CI with AO sample clock
                        ci_task.timing.cfg_samp_clk_timing(
                            rate=self.config.sample_rate,
                            source=f"/{self.config.device_name}/ao/SampleClock",
                            sample_mode=AcquisitionType.FINITE,
                            samps_per_chan = n_reloc_samples
                        )
                        
                        writer = AnalogMultiChannelWriter(ao_task.out_stream, auto_start=False)
                        number_of_samples_written_signal = ao_task.write(xy_back_signal, auto_start=False)
                        
                        # Start tasks - CI first then AO
                        ci_task.start()
                        ao_task.start()
                        # ao_task.write(xy_signal, auto_start=True)
                        ci_task.read(
                            number_of_samples_per_channel=n_reloc_samples,
                            timeout=2.0
                        )
                    logging.info("Relocación al cero completada correctamente.")
                except Exception as e:
                    logging.error(f"Error en relocación: {e}")
                    self._stop_event.set()
                    return
                logging.info("Scan completed successfully")
            elif self._stop_event.is_set():
                logging.info("Scan was interrupted")
            else:
                logging.error("Scan terminated with errors")


    
    def stop(self):
        """Gracefully stop the scan."""
        self._stop_event.set()
        self._scanning = False

 
    


class NIDAQScan(BaseScan):
    """NIDAQ-based scanner for microscopy systems."""
    
    def __init__(self, shutter_backend, config: Optional[ScannerConfig] = None):
        self.config = config or ScannerConfig()
        self._start_callbacks: List[StartCallback] = []
        self._stop_callbacks: List[StopCallback] = []
        self._line_callbacks: List[ScanCallback] = []
        self._thread: Optional[_NIDAQScanThread] = None
        self.scan_params: Optional[scan_parameters.RegionScanParameters] = None
        self.scan_data = scan_parameters.RegionScanData.FIRST
        self._stop_event = threading.Event()
        self._scanning = False
        self._lock = threading.Lock()
        self.current_position = np.array([0.0, 0.0]) 
        self.shutter_backend = shutter_backend
        
    def is_scanning(self) -> bool:
      """Thread-safe way to check scanning status."""
      with self._lock:
          return self._scanning and self._thread is not None and self._thread.is_alive()

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
        if not self.scan_params:
            raise AttributeError("scan_params not set")
        return self.scan_params.line_length_fast
    
    @property
    def sslow(self) -> float:
        if not self.scan_params:
            raise AttributeError("scan_params not set")
        return self.scan_params.line_length_slow

    def start_scan(self,  params: scan_parameters.RegionScanParameters):
        """Start a new scan with given parameters."""
        if self._scanning:
            self.stop_scan()
            
        with self._lock:
            self.scan_params = dataclasses.replace(params)
            self._stop_event.clear()
            self._scanning = True
        if not self._validate_scan_params(params):
             raise ValueError("Invalid scan parameters")
     
        self.shutter_backend._execute_shutter_start_callbacks()   
        self._execute_scan_start_callbacks()
        self._thread = self._create_scan_thread()
        self._thread.start()

    def _validate_scan_params(self, params: scan_parameters.RegionScanParameters) -> bool:
        valid = True
        if params.line_length_fast <= 0 or params.line_length_slow <= 0:
            logging.error("Scan size must be positive")
            valid = False
        if params.dwell_time < 1e-6:
            logging.error("Dwell time too small")
            valid = False
        if params.pixel_size <= 0:
            logging.error("Pixel size must be positive")
            valid = False
            
        if params.line_length_fast > 100 or params.line_length_slow > 100:
            logging.error("Region out of range")
            valid = False
        return valid


    def _execute_scan_start_callbacks(self):
        """Notify all start callbacks."""
        data_shape = self.get_data_shape()
        for callback in self._start_callbacks:
            try:
                callback(self.scan_params, data_shape)
            except Exception as e:
                logging.error(f"Start callback error: {e}")


    def stop_scan(self):
        """Stop the current scan."""
        if not self._scanning:
            logging.warning("No active scan to stop")
            return
        
        # First set stop flags
        with self._lock:
            self._stop_event.set()
            self._scanning = False
        
        # Then stop the thread
        if self._thread:
            self._thread.stop()
            
            # Wait with reasonable timeout (e.g., 2 seconds)
            self._thread.join(timeout=2.0)
            
            if self._thread.is_alive():
                logging.error("Failed to stop scan thread within timeout")
                # Force cleanup if needed
                self._thread = None
            else:
                self._thread = None
                self._execute_scan_stop_callbacks()
    

    def _execute_scan_stop_callbacks(self):
        """Notify all stop callbacks."""
        for callback in self._stop_callbacks:
            try:
                callback()
            except Exception as e:
                logging.error(f"Stop callback error: {e}")

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
        import math, sys
        self.a_max = 130 # µm/ms²
    
        # --- PARÁMETROS FIJOS (documentados en SI) ---
        # Originalmente 130 µm/ms^2 -> equivale a 130 m/s^2 (ver conversión en la explicación)
    
        # Validaciones básicas
        if pixel_size <= 0  or dwell_time <= 0:
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
        n_acc =  max(int(n_acc_min), 4)  # Asegurar mínimo 4
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
        logging.info(f"Posición actual actualizada: ({fast}, {slow}) µm")

    def _create_scan_thread(self) -> _NIDAQScanThread:
       
        # Convertir todo a metros consistentemente
        params = self.scan_params
        dwell_time = float(params.dwell_time * 1E-3)  # ms -> s
        px_size = float(params.pixel_size * 1E-6)     # µm -> m
        sfast = float(params.line_length_fast * 1E-6)       # µm -> m
        sslow = float(params.line_length_slow * 1E-6)       # µm -> m
        start_slow = float(params.start_point[1] * 1E-6) # µm -> m
        n_pix = int(params.true_px_num)
       
        required_sample_rate = 1.0 / (dwell_time)
    
        # Crear configuración temporal para el thread
        thread_config = ScannerConfig(
            device_name=self.config.device_name,
            um_to_volts_DAQ=self.config.um_to_volts_DAQ,
            um_to_volts_NANO = self.config.um_to_volts_NANO,
            ao_channels=self.config.ao_channels,
            ci_channel=self.config.ci_channel,
            sample_rate=required_sample_rate,  # Usar el nuevo sample rate
            max_voltage=self.config.max_voltage
        )
      
        chann_asign = {
                    "XY": {"fast": self.config.um_to_volts_DAQ, "slow":self.config.um_to_volts_DAQ},
                    "XZ": {"fast": self.config.um_to_volts_DAQ, "slow": self.config.um_to_volts_NANO},
                    "YZ": {"fast": self.config.um_to_volts_DAQ,"slow": self.config.um_to_volts_NANO}}
                
        fast_chan_convertion = chann_asign[self.scan_params.scan_type.value]["fast"]
        slow_chan_convertion = chann_asign[self.scan_params.scan_type.value]["slow"]
        
        # Calcular parámetros
        fast0, true_px, n_px_acc, _, v_f, acc, _ = self._calculate_parameters(
            sfast,sslow,px_size, dwell_time,n_pix
        )
       
        # generar recentrado solo si haslow center en params
        # x_0, y_0 = self.current_position
        center = getattr(params, "center", None)
        reloc_signal = None
       
        if center is not None:
            # Preferir prev_center si existe (centro anterior en µm)
            fast_f_um = copy.deepcopy(params.start_point[0]*1E-6)
            slow_f_um = copy.deepcopy(2*params.end_point[1]*1E-6)
            # x_f_um = params.start_point[0]*2
            # y_f_um =  params.start_point[1]*2
            
            
        
            t_rel, fast_rel_um, slow_rel_um, n_rel = self.move_to_center_scan(
            0.0,
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
            self.update_current_position(fast_f_um,slow_f_um)
        if sslow>sfast:
            n_lines =  int(sslow / (px_size))      
        else:
            n_lines = int(sfast / px_size)
       
        # --- Asumo: x_rel_um, y_rel_um están en metros (rename para claridad) ---
        fast_rel_m = fast_rel_um    # si realmente son metros
        slow_rel_m = slow_rel_um
        
        # Asegurar >=2 puntos (en metros)
        if fast_rel_m.size < 2:
            fast_rel_m = np.pad(fast_rel_m, (0, 2 - fast_rel_m.size), mode='edge')
        if slow_rel_m.size < 2:
            slow_rel_m = np.pad(slow_rel_m, (0, 2 - slow_rel_m.size), mode='edge')
        print(f"la cantidad de pixeles por linea es {true_px}")
        # Trajectoria de escaneo: escaneo2D_back devuelve arrays en metros (supuesto)
        t, scan_fast_m, scan_slow_m, samples_per_line = self.escaneo2D_back(
            n_lines, fast0 , params.end_point[1]*1E-6, dwell_time, n_px_acc, true_px, acc, v_f, px_size
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
        t_back, back_fast_m, back_slow_m, n_back = self.move_to_center_scan(
            0.0, 0, 0, dwell_time, a_max_fast=acc, a_max_slow=acc
        )
        self.update_current_position(0,0)
        # offset para que el primer punto de la vuelta coincida con el último punto del escaneo
        last_scan_fast_m = scan_fast_m_shifted[-1]
        last_scan_slow_m = scan_slow_m_shifted[-1]
        offset_back_fast_m = last_scan_fast_m - back_fast_m[0]
        offset_back_slow_m = last_scan_slow_m - back_slow_m[0]
        
        back_fast_m_shifted = back_fast_m + offset_back_fast_m
        back_slow_m_shifted = back_slow_m + offset_back_slow_m
        
        # --- Convertir todo a voltios UNA VEZ (V/µm * 1e6 µm/m) ---
      
        volt = 1E6
        fast_rel_v = fast_rel_m * fast_chan_convertion*volt
        slow_rel_v = slow_rel_m * slow_chan_convertion*volt
        
        volt_fast = scan_fast_m_shifted * fast_chan_convertion*volt
        volt_slow = scan_slow_m_shifted * slow_chan_convertion*volt
        
        fast_back_v = back_fast_m_shifted * fast_chan_convertion*volt
        slow_back_v = back_slow_m_shifted * slow_chan_convertion*volt
        
        
        # Clipping
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
                
               
        
       # # Generar trayectorias de frames
       #  t, volt_x, volt_y, samples_per_line = self.escaneo2D_back(
       #      n_lines, x0 , params.end_point[1]*1E-6, dwell_time, n_px_acc, true_px, acc, v_f, px_size
       #  )

       

           
         
       #  offset_x =   x_rel_v[-1]*np.ones_like(volt_x)
       #  offset_y = y_rel_v[-1]*np.ones_like(volt_y)
       #  volt_x = volt_x * self.config.um_to_volts*1E6 + offset_x
       #  volt_y = volt_y* self.config.um_to_volts*1E6 + offset_y
        
        
       #  # generar vuelta final al origen
       #  t_back, x_back, y_back, n_back = self.move_to_center_scan(
       #  0.0,
       #  0, 
       #  0,
       #  dwell_time,
       #  a_max_x=acc, 
       #  a_max_y=acc
       #              )
       #  x_back_v = x_back * self.config.um_to_volts*1E6
       #  y_back_v = y_back * self.config.um_to_volts*1E6
       #  # clip por seguridad
       #  x_back_v = np.clip(x_back_v, -self.config.max_voltage, self.config.max_voltage)
       #  y_back_v = np.clip(y_back_v, -self.config.max_voltage, self.config.max_voltage)
       #  self.update_current_position(0,0)
       #  # preparar señal XY completa para escribir (Y, X) y total_samples
      
      
       #  # Registrar información de trayectoria
       #  logging.info(f"Generated trajectory: "
       #              f"X range: {np.min(volt_x):.2f} to {np.max(volt_x):.2f} µm, "
       #              f"Y range: {np.min(volt_y):.2f} to {np.max(volt_y):.2f} µm, "
       #              f"Samples: {len(volt_x)}, Lines: {n_lines}")
        # def muestra_escaneo(titulo,t,x,y): #grafica lo que le mandamos a los espejos en voltaje
        #     fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6), sharex=True)
        #     ax1.scatter(t, x,s = 8, color ="black")
        #     ax1.set_ylabel("Posición X [V]")
        #     ax1.set_title(titulo)
        #     ax1.grid(True)
        #     ax2.scatter(t, y,s = 8, color="black")
        #     ax2.set_ylabel("Posición Y [V]")
        #     ax2.set_xlabel("Tiempo [us]")
        #     ax2.grid(True)
        #     plt.tight_layout()
        #     plt.show()
        # if len(t_rel) > 1:
        #       muestra_escaneo("Voltajes de recentrado",t_rel,fast_rel_v,slow_rel_v)
        # muestra_escaneo(f"Escaneo de {n_lines + 1} de escaneo", t, volt_fast, volt_slow)
        # if len(t_back) > 1:
        #         muestra_escaneo("Voltajes de vuelta al origen",t_back,fast_back_v,slow_back_v)
       
        # devolver/crear thread pasando solo señales slowa procesadas:
        return _NIDAQScanThread(
            params=params,
            line_callbacks=self._line_callbacks,
            volt_fast = volt_fast,
            volt_slow = volt_slow,
            fast_rel = fast_rel_v,
            slow_rel = slow_rel_v,
            fast_back_v = fast_back_v,
            slow_back_v = slow_back_v,
            samples_per_line=samples_per_line,
            total_samples=total_samples,
            line_indices=line_indices,
            total_center_samples =  total_center_samples,
            total_back_samples = total_back_samples,
            true_px=true_px,
            n_px_acc=n_px_acc,
            n_lines=n_lines,
            config=thread_config
        )
       
        print(n_px_acc)
  
        # # Crear y retornar thread
        # return _NIDAQScanThread( params = params,
        #     line_callbacks=self._line_callbacks,
        #     moving_center_func=self.finish_scan,
        #     volt_x=volt_x,
        #     volt_y=volt_y,
        #     t = t,
        #     acc_um_s2 = acc * 1e6,
        #     true_px=true_px,
        #     n_px_acc=n_px_acc,
        #     samples_per_line=samples_per_line,
        #     n_lines=n_lines,
        #     config=thread_config
        # )

    def finish_scan(self, t_0: float, fast_0: float, slow_0: float,
                  fast_f: float, slow_f: float, dwell_time: float,
                  a_max_fast: float, a_max_slow: float) -> Tuple:
        """Generate flyback trajectory."""
        dfast = abs(fast_f - fast_0)
        dslow = abs(slow_f- slow_0)
        t_fast = 2 * np.sqrt(dfast / a_max_fast)
        t_slow = 2 * np.sqrt(dslow / a_max_slow)
        
        a_fast = a_max_fast
        a_slow = a_max_slow
        t_total = max(t_fast, t_slow)
        
        t_end = t_total + t_0
        t = np.arange(t_0, t_end + dwell_time, dwell_time)
        n_points = len(t)
        t_half = t_total / 2
        fast_back = np.zeros(n_points)
        slow_back = np.zeros(n_points)
        s_fast = 1 if fast_f > fast_0 else -1
        s_slow = 1 if slow_f > slow_0 else -1
        t_rel = t - t_0
        
        # First half of movement
        mask1 = t_rel < t_half
        t1 = t_rel[mask1]
        fast_back[mask1] = fast_0 + 0.5 * s_fast * a_fast * t1**2
        slow_back[mask1] = slow_0 + 0.5 * s_slow * a_slow * t1**2
        
        # Second half of movement
        mask2 = t_rel >= t_half
        t2 = t_rel[mask2] - t_half
        v_slow = a_slow * t_half
        
        fast_back = np.full_like(t, fast_0)
        
        slow_back[mask2] = (slow_0 + 0.5 * s_slow * a_slow * t_half**2 +
                          s_slow * v_slow * t2 - 0.5 * s_slow * a_slow * t2**2)
        
        return t, fast_back, slow_back, n_points
    
    def move_to_center_scan(self, t_0,
                        fast_f: float, slow_f: float, dwell_time: float,
                        a_max_fast: float, a_max_slow: float) -> Tuple:
        """Generate flyback trajectory with independent axis timing."""
        fast_0, slow_0 = self.current_position
        dfast = abs(fast_f - fast_0)
        dslow = abs(slow_f - slow_0)
    
        # Tiempo de movimiento para cada eje
        t_fast = 2 * np.sqrt(dfast / a_max_fast) if dfast > 0 else 0.0
        t_slow = 2 * np.sqrt(dslow / a_max_slow) if dslow > 0 else 0.0
    
        # Signos de movimiento
        s_fast = np.sign(fast_f - fast_0) if dfast > 0 else 0
        s_slow = np.sign(slow_f - slow_0) if dslow > 0 else 0
    
        # Tiempo total = el más largo de ambos
        t_total = max(t_fast, t_slow)
        t_end = t_0 + t_total
        t = np.arange(t_0, t_end + dwell_time, dwell_time)
        n_points = len(t)
    
        fast_back = np.empty_like(t)
        slow_back = np.empty_like(t)
    
        # --- EJE X ---
        if s_fast == 0:
            # Sin movimiento
            fast_back[:] = fast_0
        else:
            t_half_fast = t_fast / 2
            mask_move_fast = (t - t_0) < t_fast
            mask1_fast = (t - t_0) < t_half_fast
            mask2_fast = mask_move_fast & ~mask1_fast
    
            # Aceleración
            t1 = (t - t_0)[mask1_fast]
            fast_back[mask1_fast] = fast_0 + 0.5 * s_fast * a_max_fast * t1**2
    
            # Desaceleración
            t2 = (t - t_0)[mask2_fast] - t_half_fast
            v_fast = a_max_fast * t_half_fast
            fast_back[mask2_fast] = (fast_0 + 0.5 * s_fast * a_max_fast * t_half_fast**2 +
                               s_fast * v_fast * t2 - 0.5 * s_fast * a_max_fast * t2**2)
    
            # Mantener en destino tras finalizar
            fast_back[~mask_move_fast] = fast_f
    
        # --- EJE Y ---
        if s_slow == 0:
            slow_back[:] = slow_0
        else:
            t_half_slow = t_slow / 2
            mask_move_slow = (t - t_0) < t_slow
            mask1_slow = (t - t_0) < t_half_slow
            mask2_slow = mask_move_slow & ~mask1_slow
    
            t1 = (t - t_0)[mask1_slow]
            slow_back[mask1_slow] = slow_0 + 0.5 * s_slow * a_max_slow * t1**2
    
            t2 = (t - t_0)[mask2_slow] - t_half_slow
            v_slow = a_max_slow * t_half_slow
            slow_back[mask2_slow] = (slow_0 + 0.5 * s_slow * a_max_slow * t_half_slow**2 +
                               s_slow * v_slow * t2 - 0.5 * s_slow * a_max_slow * t2**2)
    
            slow_back[~mask_move_slow] = slow_f
    
        
   
        return t, fast_back, slow_back, n_points

    def escaneo2D_back(self, n_lines: int, fast_0: float, slow_0: float,
                     dwell_time: float, n_pix_acc: int, n_pix: int,
                     acc: float, v_f: float,px_size):
        if n_pix_acc <= 0 or n_pix <= 0:
            raise ValueError("n_pix_acc and n_pix must be positive")
        
        t_line_duration = 4 * (n_pix_acc * dwell_time) + 2 * n_pix * dwell_time
        t_local = np.arange(0, t_line_duration , dwell_time)
        idx_pix = np.arange(len(t_local))
        n_points = 4 * n_pix_acc  + 2 * n_pix 
        v_slow = v_f / 2

        fast = np.zeros_like(t_local)
        slow = np.full_like(t_local, slow_0)  # Constant slow for entire line


        # Aceleración inicial
        mask1 = idx_pix < n_pix_acc
        fast[mask1] = -fast_0 + 0.5 * acc * (idx_pix[mask1] * dwell_time) ** 2

        # Velocidad constante
        mask2 = (idx_pix >= n_pix_acc) & (idx_pix <= n_pix + n_pix_acc)
        fast[mask2] = -fast_0 + 0.5 * acc * ((n_pix_acc * dwell_time) ** 2) + v_f * (idx_pix[mask2] - n_pix_acc) * dwell_time

        # Deceleración positiva
        mask3 = (idx_pix > n_pix + n_pix_acc) & (idx_pix <= n_pix + 2 * n_pix_acc)
        t_dec = (idx_pix[mask3] - (n_pix + n_pix_acc)) * dwell_time
        fast[mask3] = (-fast_0 + 0.5 * acc * ((n_pix_acc * dwell_time) ** 2) +
                    v_f * n_pix * dwell_time + v_f * t_dec - 0.5 * acc * t_dec ** 2)

        # Aceleración negativa
        mask4 = (idx_pix >= n_pix + 2 * n_pix_acc) & (idx_pix < n_pix + 3 * n_pix_acc)
        t_acc_neg = (idx_pix[mask4] - (n_pix + 2 * n_pix_acc)) * dwell_time
        fast[mask4] = fast_0 + v_f * (n_pix * dwell_time) - 0.5 * acc * (t_acc_neg) ** 2

        # Velocidad negativa
        mask5 = (idx_pix >= n_pix + 3 * n_pix_acc) & (idx_pix <= 2 * n_pix + 3 * n_pix_acc)
        fast[mask5] = (fast_0 + v_f * (n_pix * dwell_time) -
                    0.5 * acc * ((n_pix_acc * dwell_time) ** 2) -
                    v_f * (idx_pix[mask5] - (n_pix + 3 * n_pix_acc)) * dwell_time)

        # Deceleración final
        mask6 = (2 * n_pix + 4 * n_pix_acc >= idx_pix) & (idx_pix >= 2 * n_pix + 3 * n_pix_acc)
        t_dec_final = (idx_pix[mask6] - (2 * n_pix + 3 * n_pix_acc)) * dwell_time
        fast[mask6] = (fast_0 + v_f * (n_pix * dwell_time) -
                    0.5 * acc * ((n_pix_acc * dwell_time) ** 2) -
                    v_f * n_pix * dwell_time - v_f * t_dec_final +
                    0.5 * acc * t_dec_final ** 2)
      

        # N líneas de escaneo
        num_points = len(t_local) 
        t_offsets = np.arange(n_lines) * num_points * dwell_time
        t_total = np.tile(t_local, n_lines) + np.repeat(t_offsets, num_points)  

        fast_total = np.tile(fast, n_lines)
        # x_total -= (x_total.max() + x_total.min()) / 2

        slow_step =  px_size
        slow_shifts = slow_0 - np.arange(n_lines) * slow_step
        slow_offsets = np.repeat(slow_shifts, num_points)
        # slow_total = np.tile(np.ones_like(slow)*slow_0, n_lines) + slow_offsets
        slow_total = np.tile(slow, n_lines) + (slow_offsets - slow_0)
        # slow_total -= (slow_total.max() + slow_total.min()) / 2
       

        # le pido que n_lineso haga una subida mas en slow
        # slow_total[-num_points:] = slow_total[-num_points - 1]

        
        last_fast = fast_total[-1]
        last_slow = slow_total[-1]
        last_t = t_total[-1]
        
      
        
        # Garantizar que el último punto sea exactamente slow_0
        
        t_back, fast_back, slow_back,_ = self.finish_scan(last_t, last_fast, last_slow, fast_0, slow_0,dwell_time, acc, acc)

        # Concatenar la vuelta
        t_total = np.concatenate([t_total, t_back])
        fast_total = np.concatenate([fast_total, fast_back])
        slow_total = np.concatenate([slow_total, slow_back])
        # slow_total -= (slow_total.max() + slow_total.min()) / 2
        # slow_total[-num_points:] = slow_total[-num_points]
        # x_total -= (x_total.max() + x_total.min()) / 2


        return t_total, fast_total, slow_total, int(n_points)
        
      

    def get_data_shape(self) -> Tuple[int, int]:
        """Get scan data dimensions."""
        if not self.scan_params:
            return (0, 0)
        return (int(self.sfast / self.scan_params.pixel_size),
                int(self.sslow / self.scan_params.pixel_size))

    def get_extents(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get scan extents in physical units."""
        if not self.scan_params:
            return ((0, 0), (0, 0))
        return ((0, self.sfast), 
                (0, self.sslow))