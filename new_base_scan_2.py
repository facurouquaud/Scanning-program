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

# Define callback types
StartCallback = Callable[[object, Tuple[int, int]], None]
StopCallback = Callable[[], None]
ScanCallback = Callable[[np.ndarray, int, bool], None]

# Configuration dataclass
class ScannerConfig:
    def __init__(self,
                 device_name: str = "Dev1",
                 um_to_volts: float = 0.04,
                 ao_channels: List[str] = ["ao0", "ao1"],
                 ci_channel: str = "ctr0",
                 sample_rate: float = 100000.0,
                 max_voltage: float = 5):
        self.device_name = device_name
        self.um_to_volts = um_to_volts
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
    
    def __init__(self, params, line_callbacks,volt_x, volt_y,x_rel, y_rel,x_back_v, y_back_v,
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
        self.volt_x = volt_x
        self.volt_y = volt_y
        self.x_rel = x_rel
        self.y_rel = y_rel
        self.x_back_v = x_back_v
        self.y_back_v = y_back_v
        self.total_back_samples = total_back_samples
        self.true_px = true_px
     
        
        
        print(self.true_px)
        # Convert position to voltage (safe copy)
      

        # Validate voltage ranges
        max_physical = config.max_voltage / config.um_to_volts
        # self.volt_x = np.clip(volt_x.copy(), -max_physical, max_physical) * config.um_to_volts
        # self.volt_y = np.clip(volt_y.copy(), -max_physical, max_physical) * config.um_to_volts
        # self.t = t
        
        self._validate_voltage(config.max_voltage)

        # Validación adicional
        
        if np.max(np.abs(self.volt_x)) > config.max_voltage:
             actual_max = np.max(np.abs(self.volt_x))
             raise ValueError(
                 f"Voltaje X excede {config.max_voltage}V (llegó a {actual_max:.2f}V)\n"
                 f"Revisar um_to_volts (actual: {config.um_to_volts}) y tamaño de escaneo"
             )
        self.samples_per_line = samples_per_line
        self.n_lines = n_lines
        self.config = config
        self.total_samples = len(self.volt_x)
        self.total_center_samples =  total_center_samples
        # Pre-calculate line indices
        self.line_indices = [
            (i * samples_per_line, (i + 1) * samples_per_line)
            for i in range(n_lines)
        ]
  

    def _validate_voltage(self, max_voltage: float):
        """Ensure voltages are within DAQ limits with detailed logging."""
        x_min = np.min(self.volt_x)
        x_max = np.max(self.volt_x)
        y_min = np.min(self.volt_y)
        y_max = np.max(self.volt_y)
       
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
        """Main scanning loop."""
        last_line_last_pixel = None
        processed_data = []
        
        # Precompute center and flyback samples as integers
        center_samples = int(round(self.total_center_samples))
        flyback_samples = self.total_samples - center_samples - (self.n_lines * self.samples_per_line)
        
        # Ensure flyback_samples is non-negative
        flyback_samples = max(0, flyback_samples)
        back_samples = self.total_back_samples
        
        
        # AO-only relocation - Mismo enfoque que el escaneo principal
        if self.x_rel is not None and self.y_rel is not None:
            try:
                xy_reloc_signal = np.vstack((self.y_rel, self.x_rel))
                n_reloc_samples = len(self.x_rel)
                
                with nidaqmx.Task() as ao_task:
                    # Configurar canales AO igual que en el escaneo principal
                    ao_task.ao_channels.add_ao_voltage_chan("Dev1/ao1")  # Y
                    ao_task.ao_channels.add_ao_voltage_chan("Dev1/ao0")  # X
                    
                    # Configurar timing igual que en el escaneo principal
                    ao_task.timing.cfg_samp_clk_timing(
                        rate=self.config.sample_rate,
                        sample_mode=AcquisitionType.FINITE,
                        samps_per_chan=n_reloc_samples
                    )
                    
                    # Escribir datos igual que en el escaneo principal
                    writer = AnalogMultiChannelWriter(ao_task.out_stream)
                    writer.write_many_sample(xy_reloc_signal)
                    
                    # Iniciar y esperar completación
                    ao_task.start()
                    ao_task.stop()
                    
                logging.info("Relocación completada correctamente.")
            except Exception as e:
                logging.error(f"Error en relocación: {e}")
                self._stop_event.set()
                return
    
        try:
            self._scanning = True
            frame_count = 0
            
            while not self._stop_event.is_set() and self._scanning:
                xy_signal = np.vstack((self.volt_y, self.volt_x))
                
                with nidaqmx.Task() as ao_task, nidaqmx.Task() as ci_task:
                    # Configure analog output
                    ao_task.ao_channels.add_ao_voltage_chan("Dev1/ao1", name_to_assign_to_channel="Y")
                    ao_task.ao_channels.add_ao_voltage_chan("Dev1/ao0", name_to_assign_to_channel="X")
                    
                    # Export sample clock for synchronization
                    ao_task.export_signals.export_signal(
                        signal_id=nidaqmx.constants.Signal.SAMPLE_CLOCK,
                        output_terminal="/Dev1/PFI0"
                    )
                    
                    # Configure AO timing
                    ao_task.timing.cfg_samp_clk_timing(
                        rate=self.config.sample_rate,
                        sample_mode=AcquisitionType.FINITE,
                        samps_per_chan=self.total_samples
                    )
                    
                    # Configure counter input
                    ci_task.ci_channels.add_ci_count_edges_chan(
                        f"{self.config.device_name}/{self.config.ci_channel}",
                        edge=Edge.RISING
                    )
                    
                    # Sync CI with AO sample clock
                    ci_task.timing.cfg_samp_clk_timing(
                        rate=self.config.sample_rate,
                        source=f"/{self.config.device_name}/ao/SampleClock",
                        sample_mode=AcquisitionType.FINITE,
                        samps_per_chan=self.total_samples 
                    )
                    
               
                    
                    # Start tasks - CI first then AO
                    ci_task.start()
                    # time.sleep(0.2)

                    ao_task.start()
                    # ao_task.write(xy_signal, auto_start=True)
                    
                    # Read and discard center samples at start of frame
                    # if center_samples > 0:
                    #     try:
                    #         center_data = ci_task.read(
                    #             number_of_samples_per_channel=center_samples,
                    #             timeout=4.0
                    #         )
                    #     except Exception as e:
                    #         logging.error(f"Error reading center samples: {e}")
                    #         self._stop_event.set()
                    #         continue
                    
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
                            
                            # Real-time processing
                            normalized_line = current_line if last_line_last_pixel is None \
                                else current_line - last_line_last_pixel
                            
                            last_line_last_pixel = current_line[-1]  # Update for next line
                            
                            diff_line = np.insert(np.diff(normalized_line), 0, 0)
                            processed_line = diff_line
                            
                            # Store for visualization
                            processed_data.append(processed_line)
                            
                            last_line = (line_idx == self.n_lines - 1)
                            
                            # Send to callbacks
                            for callback in self._line_callbacks:
                                try:
                                    if callback(processed_line, line_idx, last_line):
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
            if not self._error_occurred and not self._stop_event.is_set():
                try:
                    xy_back_signal = np.vstack((self.y_back_v, self.x_back_v))
                    n_reloc_samples = len(self.x_back_v)
                    
                    with nidaqmx.Task() as ao_task:
                        # Configurar canales AO igual que en el escaneo principal
                        ao_task.ao_channels.add_ao_voltage_chan("Dev1/ao1")  # Y
                        ao_task.ao_channels.add_ao_voltage_chan("Dev1/ao0")  # X
                        
                        # Configurar timing igual que en el escaneo principal
                        ao_task.timing.cfg_samp_clk_timing(
                            rate=self.config.sample_rate,
                            sample_mode=AcquisitionType.FINITE,
                            samps_per_chan=back_samples
                        )
                        
                        # Escribir datos igual que en el escaneo principal
                        writer = AnalogMultiChannelWriter(ao_task.out_stream)
                        writer.write_many_sample(xy_back_signal)
                        
                        # Iniciar y esperar completación
                        ao_task.start()
                        ao_task.stop()
                        
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
    
    def __init__(self, config: Optional[ScannerConfig] = None):
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
    def sx(self) -> float:
        if not self.scan_params:
            raise AttributeError("scan_params not set")
        return self.scan_params.line_length_x
    
    @property
    def sy(self) -> float:
        if not self.scan_params:
            raise AttributeError("scan_params not set")
        return self.scan_params.line_length_y

    def start_scan(self,  params: scan_parameters.RegionScanParameters):
        """Start a new scan with given parameters."""
        if self.is_scanning:
            self.stop_scan()
            
        with self._lock:
            self.scan_params = dataclasses.replace(params)
            self._stop_event.clear()
            self._scanning = True
            
        if not self._validate_scan_params(params):
             raise ValueError("Invalid scan parameters")
            
        self._execute_scan_start_callbacks()
        self._thread = self._create_scan_thread()
        self._thread.start()

    def _validate_scan_params(self, params: scan_parameters.RegionScanParameters) -> bool:
        valid = True
        if params.line_length_x <= 0 or params.line_length_y <= 0:
            logging.error("Scan size must be positive")
            valid = False
        if params.dwell_time < 1e-6:
            logging.error("Dwell time too small")
            valid = False
        if params.pixel_size <= 0:
            logging.error("Pixel size must be positive")
            valid = False
            
        if params.line_length_x > 100 or params.line_length_y > 100:
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
        if not self.is_scanning:
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

    def _calculate_parameters(self, scan_width_x: float, scan_width_y, pixel_size, dwell_time: float, n_pix) -> Tuple:
        """Calculate scanning parameters with type safety."""
        try:
            # Validar valores positivos
            self.a_max = 130  # µm/ms² 
            if pixel_size <= 0 or self.a_max <= 0 or dwell_time <= 0:
                self._execute_scan_stop_callbacks()
                raise ValueError("Parameters must be positive")
            
            # Cálculos principales
            v_f = pixel_size / dwell_time
            t_acc_min = v_f / self.a_max
            n_acc_min = math.ceil(t_acc_min / dwell_time)
            n_acc =  max(int(n_acc_min), 4)  # Asegurar mínimo 4
            acc = v_f / (n_acc * dwell_time)
            print(f"la acc es {acc}")
            if scan_width_x > 100 or scan_width_y >100:
                self._execute_scan_stop_callbacks()

                raise ValueError("Region out of range")
            if acc > self.a_max:
                self._execute_scan_stop_callbacks()

                raise ValueError("Acceleration out of range")
            
            if n_pix > 500:
                self._execute_scan_stop_callbacks()
                raise ValueError("number of pixels out of range")
                
            x0 = (0.5 * acc * ((n_acc * dwell_time) ** 2))
            
            
            return x0, n_pix, n_acc, n_acc_min, v_f, acc, v_f
        
        except (TypeError, ValueError) as e:
            logging.error(f"Parameter calculation error: {e}")
            raise
    def update_current_position(self, x: float, y: float):
        """Actualiza la posición actual del escáner."""
        self.current_position = np.array([x, y])
        logging.info(f"Posición actual actualizada: ({x}, {y}) µm")

    def _create_scan_thread(self) -> _NIDAQScanThread:
        """Create and configure scan thread with voltage clamping."""
        # Convertir todo a metros consistentemente
        params = self.scan_params
        dwell_time = float(params.dwell_time * 1E-3)  # ms -> s
        px_size = float(params.pixel_size * 1E-6)     # µm -> m
        sx = float(params.line_length_x * 1E-6)       # µm -> m
        sy = float(params.line_length_y * 1E-6)       # µm -> m
        start_y = float(params.start_point[1] * 1E-6) # µm -> m
        n_pix = int(params.true_px_num)
       
        required_sample_rate = 1.0 / (dwell_time)
    
        # Crear configuración temporal para el thread
        thread_config = ScannerConfig(
            device_name=self.config.device_name,
            um_to_volts=self.config.um_to_volts,
            ao_channels=self.config.ao_channels,
            ci_channel=self.config.ci_channel,
            sample_rate=required_sample_rate,  # Usar el nuevo sample rate
            max_voltage=self.config.max_voltage
        )
      
        
        
        
        # Calcular parámetros
        x0, true_px, n_px_acc, _, v_f, acc, _ = self._calculate_parameters(
            sx,sy,px_size, dwell_time,n_pix
        )
        print(x0)
       
        # generar recentrado solo si hay center en params
        # x_0, y_0 = self.current_position
        center = getattr(params, "center", None)
        reloc_signal = None
       
        if center is not None:
            # Preferir prev_center si existe (centro anterior en µm)
            x_f_um = np.copy(-x0*1E6 + params.start_point[0])
            y_f_um = np.copy(-start_y*1E6)
            # x_f_um = params.start_point[0]*2
            # y_f_um =  params.start_point[1]*2
            
            
        
            t_rel, x_rel_um, y_rel_um, n_rel = self.move_to_center_scan(
            0.0,
            x_f_um, 
            y_f_um,
            dwell_time,
            a_max_x=acc, 
            a_max_y=acc
                        )
                # convertir a V (si config.um_to_volts es V/µm)
            x_rel_v = x_rel_um * self.config.um_to_volts
            y_rel_v = y_rel_um * self.config.um_to_volts
            print(x_rel_v)
            print(y_rel_v)
            
        
            # asegurar al menos 2 puntos
            if x_rel_v.size < 2:
                x_rel_v = np.pad(x_rel_v, (0, 2 - x_rel_v.size), mode='edge')
            if y_rel_v.size < 2:
                y_rel_v = np.pad(y_rel_v, (0, 2 - y_rel_v.size), mode='edge')
        
            # clip por seguridad
            x_rel_v = np.clip(x_rel_v, -self.config.max_voltage, self.config.max_voltage)
            y_rel_v = np.clip(y_rel_v, -self.config.max_voltage, self.config.max_voltage)
            self.update_current_position(x_f_um,y_f_um)
        if sy>sx:
            n_lines =  int(sy / (px_size))      
        else:
            n_lines = int(sx / px_size)
       
        
       
        
       
        
       # Generar trayectorias de frames
        t, volt_x, volt_y, samples_per_line = self.escaneo2D_back(
            n_lines, x0 , start_y , dwell_time, n_px_acc, true_px, acc, v_f,px_size
        )

       

           
         
        offset_x =   x_rel_v[-1]*np.ones_like(volt_x)
        offset_y = y_rel_v[-1]*np.ones_like(volt_y)
        volt_x = volt_x * self.config.um_to_volts*1E6 + offset_x
        volt_y = volt_y* self.config.um_to_volts*1E6 + offset_y
        
        
        # generar vuelta final al origen
        t_back, x_back, y_back, n_back = self.move_to_center_scan(
        0.0,
        0, 
        0,
        dwell_time,
        a_max_x=acc, 
        a_max_y=acc
                    )
        x_back_v = x_back * self.config.um_to_volts
        y_back_v = y_back * self.config.um_to_volts
        # clip por seguridad
        x_back_v = np.clip(x_back_v, -self.config.max_voltage, self.config.max_voltage)
        y_back_v = np.clip(y_back_v, -self.config.max_voltage, self.config.max_voltage)
        self.update_current_position(0,0)
        # preparar señal XY completa para escribir (Y, X) y total_samples
        total_samples = len(volt_x)
        total_center_samples = len(x_rel_v)
        total_back_samples = len(x_back_v)
        
        # construir line_indices igual que antes
        line_indices = [(i * samples_per_line, (i + 1) * samples_per_line) for i in range(n_lines)]
        # Registrar información de trayectoria
        logging.info(f"Generated trajectory: "
                    f"X range: {np.min(volt_x):.2f} to {np.max(volt_x):.2f} µm, "
                    f"Y range: {np.min(volt_y):.2f} to {np.max(volt_y):.2f} µm, "
                    f"Samples: {len(volt_x)}, Lines: {n_lines}")
        def muestra_escaneo(titulo,t,x,y): #grafica lo que le mandamos a los espejos en voltaje
            fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6), sharex=True)
            ax1.scatter(t, x,s = 8, color ="black")
            ax1.set_ylabel("Posición X [V]")
            ax1.set_title(titulo)
            ax1.grid(True)
            ax2.scatter(t, y,s = 8, color="black")
            ax2.set_ylabel("Posición Y [V]")
            ax2.set_xlabel("Tiempo [us]")
            ax2.grid(True)
            plt.tight_layout()
            plt.show()
        if len(t_rel) > 1:
              muestra_escaneo("Voltajes de recentrado",t_rel,x_rel_v,y_rel_v)
        muestra_escaneo(f"Escaneo de {n_lines + 1} de escaneo", t, volt_x, volt_y)
        if len(t_back) > 1:
               muestra_escaneo("Voltajes de vuelta al origen",t_back,x_back_v,y_back_v)
       
        # devolver/crear thread pasando solo señales ya procesadas:
        return _NIDAQScanThread(
            params=params,
            line_callbacks=self._line_callbacks,
            volt_x = volt_x,
            volt_y = volt_y,
            x_rel = x_rel_v,
            y_rel = y_rel_v,
            x_back_v = x_back_v,
            y_back_v = y_back_v,
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

    def finish_scan(self, t_0: float, x_0: float, y_0: float,
                  x_f: float, y_f: float, dwell_time: float,
                  a_max_x: float, a_max_y: float) -> Tuple:
        """Generate flyback trajectory."""
        dx = abs(x_f - x_0)
        dy = abs(y_f- y_0)
        t_x = 2 * np.sqrt(dx / a_max_x)
        t_y = 2 * np.sqrt(dy / a_max_y)
        
        a_x = a_max_x
        a_y = a_max_y
        t_total = max(t_x, t_y)
        
        t_end = t_total + t_0
        t = np.arange(t_0, t_end + dwell_time, dwell_time)
        n_points = len(t)
        t_half = t_total / 2
        x_back = np.zeros(n_points)
        y_back = np.zeros(n_points)
        s_x = 1 if x_f > x_0 else -1
        s_y = 1 if y_f > y_0 else -1
        t_rel = t - t_0
        
        # First half of movement
        mask1 = t_rel < t_half
        t1 = t_rel[mask1]
        x_back[mask1] = x_0 + 0.5 * s_x * a_x * t1**2
        y_back[mask1] = y_0 + 0.5 * s_y * a_y * t1**2
        
        # Second half of movement
        mask2 = t_rel >= t_half
        t2 = t_rel[mask2] - t_half
        v_y = a_y * t_half
        
        x_back = np.full_like(t, x_0)
        
        y_back[mask2] = (y_0 + 0.5 * s_y * a_y * t_half**2 +
                          s_y * v_y * t2 - 0.5 * s_y * a_y * t2**2)
        
        return t, x_back, y_back, n_points
    
    def move_to_center_scan(self, t_0,
                        x_f: float, y_f: float, dwell_time: float,
                        a_max_x: float, a_max_y: float) -> Tuple:
        """Generate flyback trajectory with independent axis timing."""
        x_0, y_0 = self.current_position
        dx = abs(x_f - x_0)
        dy = abs(y_f - y_0)
    
        # Tiempo de movimiento para cada eje
        t_x = 2 * np.sqrt(dx / a_max_x) if dx > 0 else 0.0
        t_y = 2 * np.sqrt(dy / a_max_y) if dy > 0 else 0.0
    
        # Signos de movimiento
        s_x = np.sign(x_f - x_0) if dx > 0 else 0
        s_y = np.sign(y_f - y_0) if dy > 0 else 0
    
        # Tiempo total = el más largo de ambos
        t_total = max(t_x, t_y)
        t_end = t_0 + t_total
        t = np.arange(t_0, t_end + dwell_time, dwell_time)
        n_points = len(t)
    
        x_back = np.empty_like(t)
        y_back = np.empty_like(t)
    
        # --- EJE X ---
        if s_x == 0:
            # Sin movimiento
            x_back[:] = x_0
        else:
            t_half_x = t_x / 2
            mask_move_x = (t - t_0) < t_x
            mask1_x = (t - t_0) < t_half_x
            mask2_x = mask_move_x & ~mask1_x
    
            # Aceleración
            t1 = (t - t_0)[mask1_x]
            x_back[mask1_x] = x_0 + 0.5 * s_x * a_max_x * t1**2
    
            # Desaceleración
            t2 = (t - t_0)[mask2_x] - t_half_x
            v_x = a_max_x * t_half_x
            x_back[mask2_x] = (x_0 + 0.5 * s_x * a_max_x * t_half_x**2 +
                               s_x * v_x * t2 - 0.5 * s_x * a_max_x * t2**2)
    
            # Mantener en destino tras finalizar
            x_back[~mask_move_x] = x_f
    
        # --- EJE Y ---
        if s_y == 0:
            y_back[:] = y_0
        else:
            t_half_y = t_y / 2
            mask_move_y = (t - t_0) < t_y
            mask1_y = (t - t_0) < t_half_y
            mask2_y = mask_move_y & ~mask1_y
    
            t1 = (t - t_0)[mask1_y]
            y_back[mask1_y] = y_0 + 0.5 * s_y * a_max_y * t1**2
    
            t2 = (t - t_0)[mask2_y] - t_half_y
            v_y = a_max_y * t_half_y
            y_back[mask2_y] = (y_0 + 0.5 * s_y * a_max_y * t_half_y**2 +
                               s_y * v_y * t2 - 0.5 * s_y * a_max_y * t2**2)
    
            y_back[~mask_move_y] = y_f
    
        
   
        return t, x_back, y_back, n_points

    def escaneo2D_back(self, n_lines: int, x_0: float, y_0: float,
                     dwell_time: float, n_pix_acc: int, n_pix: int,
                     acc: float, v_f: float,px_size):
        if n_pix_acc <= 0 or n_pix <= 0:
            raise ValueError("n_pix_acc and n_pix must be positive")
        
        t_line_duration = 4 * (n_pix_acc * dwell_time) + 2 * n_pix * dwell_time
        t_local = np.arange(0, t_line_duration , dwell_time)
        idx_pix = np.arange(len(t_local))
        n_points = 4 * n_pix_acc  + 2 * n_pix 
        v_y = v_f / 2

        x = np.zeros_like(t_local)
        y = np.full_like(t_local, y_0)  # Constant Y for entire line


        # Aceleración inicial
        mask1 = idx_pix < n_pix_acc
        x[mask1] = -x_0 + 0.5 * acc * (idx_pix[mask1] * dwell_time) ** 2

        # Velocidad constante
        mask2 = (idx_pix >= n_pix_acc) & (idx_pix <= n_pix + n_pix_acc)
        x[mask2] = -x_0 + 0.5 * acc * ((n_pix_acc * dwell_time) ** 2) + v_f * (idx_pix[mask2] - n_pix_acc) * dwell_time

        # Deceleración positiva
        mask3 = (idx_pix > n_pix + n_pix_acc) & (idx_pix <= n_pix + 2 * n_pix_acc)
        t_dec = (idx_pix[mask3] - (n_pix + n_pix_acc)) * dwell_time
        x[mask3] = (-x_0 + 0.5 * acc * ((n_pix_acc * dwell_time) ** 2) +
                    v_f * n_pix * dwell_time + v_f * t_dec - 0.5 * acc * t_dec ** 2)

        # Aceleración negativa
        mask4 = (idx_pix >= n_pix + 2 * n_pix_acc) & (idx_pix < n_pix + 3 * n_pix_acc)
        t_acc_neg = (idx_pix[mask4] - (n_pix + 2 * n_pix_acc)) * dwell_time
        x[mask4] = x_0 + v_f * (n_pix * dwell_time) - 0.5 * acc * (t_acc_neg) ** 2

        # Velocidad negativa
        mask5 = (idx_pix >= n_pix + 3 * n_pix_acc) & (idx_pix <= 2 * n_pix + 3 * n_pix_acc)
        x[mask5] = (x_0 + v_f * (n_pix * dwell_time) -
                    0.5 * acc * ((n_pix_acc * dwell_time) ** 2) -
                    v_f * (idx_pix[mask5] - (n_pix + 3 * n_pix_acc)) * dwell_time)

        # Deceleración final
        mask6 = (2 * n_pix + 4 * n_pix_acc >= idx_pix) & (idx_pix >= 2 * n_pix + 3 * n_pix_acc)
        t_dec_final = (idx_pix[mask6] - (2 * n_pix + 3 * n_pix_acc)) * dwell_time
        x[mask6] = (x_0 + v_f * (n_pix * dwell_time) -
                    0.5 * acc * ((n_pix_acc * dwell_time) ** 2) -
                    v_f * n_pix * dwell_time - v_f * t_dec_final +
                    0.5 * acc * t_dec_final ** 2)
      

        # N líneas de escaneo
        num_points = len(t_local) 
        t_offsets = np.arange(n_lines) * num_points * dwell_time
        t_total = np.tile(t_local, n_lines) + np.repeat(t_offsets, num_points)  

        x_total = np.tile(x, n_lines)
        # x_total -= (x_total.max() + x_total.min()) / 2

        y_step =  px_size
        y_shifts = y_0 - np.arange(n_lines) * y_step
        y_offsets = np.repeat(y_shifts, num_points)
        # y_total = np.tile(np.ones_like(y)*y_0, n_lines) + y_offsets
        y_total = np.tile(y, n_lines) + (y_offsets - y_0)
        # y_total -= (y_total.max() + y_total.min()) / 2
       

        # le pido que n_lineso haga una subida mas en y
        # y_total[-num_points:] = y_total[-num_points - 1]

        
        last_x = x_total[-1]
        last_y = y_total[-1]
        last_t = t_total[-1]
        
      
        
        # Garantizar que el último punto sea exactamente y_0
        
        t_back, x_back, y_back,_ = self.finish_scan(last_t, last_x, last_y, x_0, y_0,dwell_time, acc, acc)

        # Concatenar la vuelta
        t_total = np.concatenate([t_total, t_back])
        x_total = np.concatenate([x_total, x_back])
        y_total = np.concatenate([y_total, y_back])
        y_total -= (y_total.max() + y_total.min()) / 2
        # y_total[-num_points:] = y_total[-num_points]
        x_total -= (x_total.max() + x_total.min()) / 2


        return t_total, x_total, y_total, int(n_points)
        
      

    def get_data_shape(self) -> Tuple[int, int]:
        """Get scan data dimensions."""
        if not self.scan_params:
            return (0, 0)
        return (int(self.sx / self.scan_params.pixel_size),
                int(self.sy / self.scan_params.pixel_size))

    def get_extents(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get scan extents in physical units."""
        if not self.scan_params:
            return ((0, 0), (0, 0))
        return ((0, self.sx), 
                (0, self.sy))