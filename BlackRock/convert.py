import os
import numpy as np
from neo.rawio import BlackrockRawIO

# --- Configuración de Archivos ---
input_file = r"C:\Users\laboratorio\Downloads\20230522-090311-001.ns2"
output_file = r"continuous_32ch.dat"

# --- Parámetros de Canales ---
n_channels_ephys = 32      # Conservar solo los primeros 32 canales

print(f"Abriendo archivo Blackrock con NeoRawIO: {input_file}...")
reader = BlackrockRawIO(filename=input_file)
reader.parse_header()

# El .ns6 corresponde al stream de alta frecuencia (30kHz)
stream_index = 0
total_samples = reader.get_signal_size(block_index=0, seg_index=0, stream_index=stream_index)
fs = reader.get_signal_sampling_rate(stream_index=stream_index)

print(f"-> Frecuencia de muestreo detectada: {fs} Hz")
print(f"-> Total de muestras en el archivo: {total_samples}")

# Tamaño del bloque para cuidar la memoria RAM (1,000,000 de muestras por iteración)
chunk_size = 1000000 
print(f"\nConvirtiendo y guardando en '{output_file}'...")

# Abrimos el archivo .dat en modo de escritura binaria pura ('wb')
with open(output_file, 'wb') as f:
    for start_idx in range(0, total_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, total_samples)
        
        # 1. Leer fragmento crudo (retorna nativamente en int16 y forma [Muestras, Canales])
        chunk = reader.get_analogsignal_chunk(
            block_index=0, 
            seg_index=0, 
            i_start=start_idx, 
            i_stop=end_idx, 
            stream_index=stream_index
        )
        
        # 2. Recortar la matriz para conservar solo los 32 canales electrofisiológicos
        chunk_ephys = chunk[:, :n_channels_ephys]
        
        # 3. Volcar los bytes directos al disco duro sin sobrecargar la RAM
        f.write(chunk_ephys.tobytes())
        
        progress = (end_idx / total_samples) * 100
        print(f"Progreso: {progress:.1f}% ({end_idx}/{total_samples} muestras procesadas)", end='\r')

print("\n" + "="*50)
print("¡Conversión exitosa!")
print(f"Archivo binario plano guardado en: {output_file}")
print("="*50)