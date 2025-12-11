# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 09:39:18 2025

@author: Lenovo
"""

# def channel_configuration_ai(mode,ao_task, ai_task,
#              xy_signal, number_of_points, slow_chan, fast_chan):
#     ao_task.ao_channels.add_ao_voltage_chan(slow_chan)  # slow
#     ao_task.ao_channels.add_ao_voltage_chan(fast_chan)  # fast
#     ao_task.timing.cfg_samp_clk_timing(
#         rate=self.config.sample_rate,
#         sample_mode=mode,
#         samps_per_chan= number_of_points
#     )
    
#     writer = AnalogMultiChannelWriter(ao_task.out_stream, auto_start=False)
#     number_of_samples_written_signal = ao_task.write(xy_signal, auto_start=False)
# from nidaqmx.stream_readers import AnalogMultiChannelReader
# from nidaqmx.constants import TerminalConfiguration

def aoai_channel_configuration(self, mode, ao_task, ai_task, xy_signal, number_of_points, slow_chan, fast_chan):
    ao_task.ao_channels.add_ao_voltage_chan(slow_chan)  # slow
    ao_task.ao_channels.add_ao_voltage_chan(fast_chan)  # fast

    # Export sample clock for synchronization (igual que antes)
    ao_task.export_signals.export_signal(
        signal_id=nidaqmx.constants.Signal.SAMPLE_CLOCK,
        output_terminal=f"/{self.config.device_name}/PFI0"
    )

    # Configure AO timing (igual que antes)
    ao_task.timing.cfg_samp_clk_timing(
        rate=self.config.sample_rate,
        sample_mode=mode,
        samps_per_chan=number_of_points
    )

    # --- AI (entrada de voltaje): configurar canales AI ---
    # TerminalConfiguration puede ser RSE, NRSE o DIFFERENTIAL según el cableado.
    ai_task.ai_channels.add_ai_voltage_chan(
        f"{self.config.device_name}/{self.config.ai_channel}",
        terminal_config=TerminalConfiguration.RSE)


    # Sincronizar el reloj de muestreo del AI con el AO SampleClock
    ai_task.timing.cfg_samp_clk_timing(
        rate=self.config.sample_rate,
        source=f"/{self.config.device_name}/ao/SampleClock",
        sample_mode=mode,
        samps_per_chan=number_of_points
    )

    # Preparar reader para leer múltiples canales desde el stream del AI
    reader = AnalogMultiChannelReader(ai_task.in_stream)

    # Preparar writer para AO (mantener la escritura del pattern xy_signal)
    writer = AnalogMultiChannelWriter(ao_task.out_stream, auto_start=False)
    number_of_samples_written_signal = ao_task.write(xy_signal, auto_start=False)

    # Devolver objetos útiles para el código que controlará la adquisición/lectura
    return {
        "ao_writer": writer,
        "ai_reader": reader,
        "samples_to_write": number_of_samples_written_signal
    }


