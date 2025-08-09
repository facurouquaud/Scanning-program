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
    
    def __init__(self,
                 params: scan_parameters.RegionScanParameters,
                 line_callbacks: List[ScanCallback],
                 volt_x: np.ndarray,
                 volt_y: np.ndarray,
                 t: np.ndarray,
                 true_px: int,
                 n_px_acc: int,
                 samples_per_line: int,
                 n_lines: int,
                 config: ScannerConfig,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scan_params = params
        self._line_callbacks = line_callbacks
        self._stop_event = threading.Event()
        self._error_occurred = False
        self.n_px_acc = n_px_acc
        self.true_px = true_px
        print(self.true_px)
        # Convert position to voltage (safe copy)
      

        # Validate voltage ranges
        max_physical = config.max_voltage / config.um_to_volts
        self.volt_x = np.clip(volt_x.copy(), -max_physical, max_physical) * config.um_to_volts
        self.volt_y = np.clip(volt_y.copy(), -max_physical, max_physical) * config.um_to_volts
        self.t = t
        
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
        
        # Pre-calculate line indices
        self.line_indices = [
            (i * samples_per_line, (i + 1) * samples_per_line)
            for i in range(n_lines)
        ]
        def muestra_escaneo(N,t,x,y): #grafica lo que le mandamos a los espejos en voltaje
            fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6), sharex=True)
            ax1.scatter(t, x,s = 8, color ="black")
            ax1.set_ylabel("Posición X [V]")
            ax1.set_title(f"Trayectoria de {N} líneas de escaneo en x e y")
            ax1.grid(True)
            ax2.scatter(t, y,s = 8, color="black")
            ax2.set_ylabel("Posición Y [V]")
            ax2.set_xlabel("Tiempo [us]")
            ax2.grid(True)
            plt.tight_layout()
            plt.show()
        muestra_escaneo(n_lines, self.t, self.volt_x, self.volt_y)

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
        last_line_last_pixel = None  # Último píxel de la línea anterior
        processed_data = []  # Almacenamiento para datos procesados (opcional)
        # Precompute flyback samples
        flyback_samples = self.total_samples - (self.n_lines * self.samples_per_line)
        try:
            self._scanning = True
            frame_count = 0
            while not self._stop_event.is_set() and self._scanning:
                # Create XY signal stack
                xy_signal = np.vstack((self.volt_y, self.volt_x))
        
                with nidaqmx.Task() as ao_task, nidaqmx.Task() as ci_task:
                    # Configure analog output
                    ao_task.ao_channels.add_ao_voltage_chan("Dev1/ao1", name_to_assign_to_channel="Y")
                    ao_task.ao_channels.add_ao_voltage_chan("Dev1/ao0", name_to_assign_to_channel="X")
                    #marker 
                    ao_task.export_signals.export_signal(
                        signal_id=nidaqmx.constants.Signal.SAMPLE_CLOCK,
                        output_terminal="/Dev1/PFI0")
    
                    # Configure timing
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
    
                  #   ci_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                  #       trigger_source=f"/{self.config.device_name}/ao/StartTrigger"
                  # )
                
                # #     # --- SYNC FIX 2: Tamaño de búfer para CI ---
                #     ci_task.in_stream.input_buf_size = self.total_samples * 2  
                    
                
        
    
                    # Write and start analog output
                    writer = AnalogMultiChannelWriter(ao_task.out_stream, auto_start=False)
                    number_of_samples_written_y = ao_task.write(xy_signal, auto_start=False)
                 #   writer.write_many_sample(xy_signal)
                  #  writer.write_many_sample(xy_signal)
                    # ci_task.start()  
                    ci_task.start()
                    ao_task.start()                

        
                    # Process line by line
                    for line_idx, (start, end) in enumerate(self.line_indices):
                        if self._stop_event.is_set():
                            logging.info("Scan stopped by user request")
                            self._scan_completed = False
                            break
        
                        # Read data from DAQ
                        try:
                            line_total_data = ci_task.read(
                                number_of_samples_per_channel=self.samples_per_line,
                           timeout = 4 )
                            # print(line_total_data)
                        except Exception as e:
                            logging.error(f"Error al leer del DAQ en línea {line_idx}: {e}")
                            self._stop_event.set()
                            break
            
                        # Check data length
                        if len(line_total_data) != self.samples_per_line:
                            logging.warning(f"Longitud incorrecta en línea {line_idx}: "
                                          f"esperada {self.samples_per_line}, "
                                          f"recibida {len(line_total_data)}")
                            continue
        
                        try:
                            # Process data
                            line_data_both, line_data_first, line_data_second = self.pixel_filter(
                                line_total_data,
                                self.true_px,
                                self.n_px_acc
                            )
        
                            last_line = (line_idx == self.n_lines - 1)
        
                            # Select data according to scan mode
                            
                            print(self.scan_params.scan_data.name)
                            # Selecciona los datos según el modo de escaneo
                            if self.scan_params.scan_data.name == 'FIRST':
                                current_line = line_data_first
                            elif self.scan_params.scan_data.name == 'SECOND':
                                current_line = line_data_second
                            else:  # 'BOTH' o por defecto
                                current_line = line_data_both
                
                            # --- PROCESAMIENTO EN TIEMPO REAL AQUÍ ---
                            # 1. Normalización entre líneas
                            if last_line_last_pixel is not None:
                                normalized_line = current_line - last_line_last_pixel
                            else:
                                normalized_line = current_line  # Primera línea sin normalizar
                            
                            # 2. Actualizar último píxel para la siguiente línea (ANTES de diferencias)
                            last_line_last_pixel = current_line[-1]
                            
                            # 3. Diferencias intrapíxel
                            diff_line = np.insert(np.diff(normalized_line), 0, 0)  # Insertar 0 al comienzo para mantener el largo
                            
                            # --- FIN DEL PROCESAMIENTO ---
                            processed_line = diff_line  # Usar esta línea procesada
                            print(len(processed_line))
                            
                            # Almacenar para visualización completa (opcional)
                            processed_data.append(processed_line)
                            
                            last_line = (line_idx == self.n_lines - 1)
                                                    
                            # Enviar datos procesados a los callbacks
                            for callback in self._line_callbacks:
                                try:
                                    should_stop = callback(processed_line, line_idx, last_line)
                                    if should_stop:
                                        self._stop_event.set()
                                except Exception as e:
                                    logging.error(f"Error en callback para línea {line_idx}: {e}")
        
                        except ValueError as e:
                            logging.error(f"Error al procesar línea {line_idx}: {e}")
                            continue
                        except Exception as e:
                            logging.error(f"Error inesperado en línea {line_idx}: {e}")
                            self._stop_event.set()
                            break
                    # Read and discard flyback samples at end of frame
                    if flyback_samples > 0 and not self._stop_event.is_set():
                         try:
                             ci_task.read(
                                 number_of_samples_per_channel=flyback_samples
                             )
                         except Exception as e:
                             logging.warning(f"Flyback read skipped: {e}")
         
                    #End of frame
                    frame_count += 1
                    logging.info(f"Completed frame {frame_count}")
                    
                    # Reset for next frame
                    last_line_last_pixel = None
            
                    
        except Exception as e:
            logging.error(f"Critical scan error: {e}")
            self._error_occurred = True
            self._scanning = False
        
        finally:
          # self._cleanup_tasks()
          if not self._error_occurred and not self._stop_event.is_set():
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

    def _calculate_parameters(self, scan_width_x: float, scan_width_y, pixel_size, dwell_time: float) -> Tuple:
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
            if scan_width_x > scan_width_y:
                n_pix = int(round(scan_width_x / pixel_size))
            else:
                n_pix = int(round(scan_width_y / pixel_size))
            if n_pix > 500:
                self._execute_scan_stop_callbacks()
                raise ValueError("number of pixels out of range")
                
            x0 = (0.5 * acc * ((n_acc * dwell_time) ** 2))
            
            
            return x0, n_pix, n_acc, n_acc_min, v_f, acc, v_f
        
        except (TypeError, ValueError) as e:
            logging.error(f"Parameter calculation error: {e}")
            raise

    def _create_scan_thread(self) -> _NIDAQScanThread:
        """Create and configure scan thread with voltage clamping."""
        # Convertir todo a metros consistentemente
        params = self.scan_params
        dwell_time = float(params.dwell_time * 1E-3)  # ms -> s
        px_size = float(params.pixel_size * 1E-6)     # µm -> m
        sx = float(params.line_length_x * 1E-6)       # µm -> m
        sy = float(params.line_length_y * 1E-6)       # µm -> m
        start_y = float(params.start_point[1] * 1E-6) # µm -> m
       
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
            sx,sy,px_size, dwell_time
        )
        print(x0)
        if sy>sx:
            n_lines =  int(sy / (px_size))      
        else:
            n_lines = int(sx / px_size)
        # Generar trayectorias
        t, volt_x, volt_y, samples_per_line = self.escaneo2D_back(
            n_lines, x0, start_y, dwell_time, n_px_acc, true_px, acc, v_f,px_size
        )
        volt_x *= 1E6
        volt_y *= 1E6
       
        print(n_px_acc)
        # Registrar información de trayectoria
        logging.info(f"Generated trajectory: "
                    f"X range: {np.min(volt_x):.2f} to {np.max(volt_x):.2f} µm, "
                    f"Y range: {np.min(volt_y):.2f} to {np.max(volt_y):.2f} µm, "
                    f"Samples: {len(volt_x)}, Lines: {n_lines}")
        
        # Crear y retornar thread
        return _NIDAQScanThread( params = params,
            line_callbacks=self._line_callbacks,
            volt_x=volt_x,
            volt_y=volt_y,
            t = t,
            true_px=true_px,
            n_px_acc=n_px_acc,
            samples_per_line=samples_per_line,
            n_lines=n_lines,
            config=thread_config
        )

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
        v_x = a_x * t_half
        v_y = a_y * t_half
        # x_back[mask2] = (x_0 + 0.5 * s_x * a_x * t_half**2 +
                          # s_x * v_x * t2 - 0.5 * s_x * a_x * t2**2)
        x_back = np.full_like(t, x_0)
        y_back[mask2] = (y_0 + 0.5 * s_y * a_y * t_half**2 +
                          s_y * v_y * t2 - 0.5 * s_y * a_y * t2**2)
        
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
        y_shifts = y_0 + np.arange(n_lines) * y_step
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