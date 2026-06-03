import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- Configuración (Basada en tus datos) ---
filename = r"C:\Users\laboratorio\Documents\GitHub\LFP-extraction-electrophysiology\LFP_output\output_file.lfp"
n_channels = 67
fs_lfp = 1250  # Frecuencia de muestreo destino
dtype = 'int16' # El estándar en electrofisiología

# 1. Cargar el archivo binario
data = np.fromfile(filename, dtype=dtype)

# 2. Reshape: Los datos binarios se guardan como [sample1_ch1, sample1_ch2, ..., sampleN_ch67]
# Por lo tanto, debemos redimensionar la matriz
n_samples = len(data) // n_channels
data = data.reshape((n_samples, n_channels))

print(f"Forma de la matriz (muestras, canales): {data.shape}")
print(f"Duración total: {n_samples / fs_lfp:.2f} segundos")

# 3. Visualización rápida de los primeros 2 segundos de los primeros 5 canales
time = np.arange(n_samples) / fs_lfp
t_start, t_stop = 4, 6
idx_start, idx_stop = int(t_start * fs_lfp), int(t_stop * fs_lfp)

plt.figure(figsize=(12, 6))
for i in range(5): # Graficamos solo 5 canales para no saturar
    plt.plot(time[idx_start:idx_stop], data[idx_start:idx_stop, i] + (i * 500)) # Offset para separar

plt.title("LFP (Canales 0-4)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (uV + offset)")
plt.grid(True)
plt.legend([f'Canal {i}' for i in range(5)], loc='upper right')
plt.xlim(t_start, t_stop)
plt.show()

# Calcular el PSD para el canal 20 (el que analizaste antes)
ch = 4
frequencies, psd = signal.welch(data[:, ch], fs_lfp, nperseg=fs_lfp*2)

plt.figure(figsize=(8, 5))
plt.semilogy(frequencies, psd)
plt.title(f"Densidad Espectral de Potencia - Canal {ch}")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("PSD (V^2/Hz)")
plt.xlim(0, fs_lfp/2)
plt.grid(True)
plt.show()

import pywt
from scipy.fft import fft, fftfreq
from scipy import signal

min_freq, max_freq = 4, 30
step_freq = 0.1
num_freqs_desired = int((max_freq - min_freq) / step_freq) + 1
freqs = np.linspace(min_freq, max_freq, num_freqs_desired)

dt = 1 / fs_lfp
wav = pywt.ContinuousWavelet('cmor1.0-1.0')
central_freq = pywt.central_frequency(wav)
scales = central_freq / (freqs * dt)

W, _ = pywt.cwt(data[idx_start:idx_stop, ch], scales, wav, method='fft', sampling_period=dt)
cwt_freqs = pywt.scale2frequency(wav, scales) / dt

W = np.abs(W)**2
segment_time = time[idx_start:idx_stop]

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(segment_time, data[idx_start:idx_stop, ch])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(t_start, t_stop)
plt.title(f'Channel {ch}')

plt.subplot(2, 1, 2)
plt.imshow(W, 
            extent=[segment_time[0], segment_time[-1], cwt_freqs.min(), cwt_freqs.max()],
            aspect='auto', cmap='jet', vmax=np.percentile(W, 99), vmin=0,
            origin='lower')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Wavelet Transform')
plt.ylim(min_freq, max_freq)
plt.xlim(t_start, t_stop)
#plt.colorbar(label='Power')
plt.tight_layout()
plt.show()

import pywt
from scipy.fft import fft, fftfreq
from scipy import signal

min_freq, max_freq = 80, 250
step_freq = 0.2
num_freqs_desired = int((max_freq - min_freq) / step_freq) + 1
freqs = np.linspace(min_freq, max_freq, num_freqs_desired)

dt = 1 / fs_lfp
wav = pywt.ContinuousWavelet('cmor2.0-2.0')
central_freq = pywt.central_frequency(wav)
scales = central_freq / (freqs * dt)

W, _ = pywt.cwt(data[idx_start:idx_stop, ch], scales, wav, method='fft', sampling_period=dt)
cwt_freqs = pywt.scale2frequency(wav, scales) / dt

W = np.abs(W)**2
segment_time = time[idx_start:idx_stop]

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(segment_time, data[idx_start:idx_stop, ch])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(t_start, t_stop)
plt.title(f'Channel {ch}')

plt.subplot(2, 1, 2)
plt.imshow(W, 
            extent=[segment_time[0], segment_time[-1], cwt_freqs.min(), cwt_freqs.max()],
            aspect='auto', cmap='jet', vmax=np.percentile(W, 99), vmin=0,
            origin='lower')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Wavelet Transform')
plt.ylim(min_freq, max_freq)
plt.xlim(t_start, t_stop)
#plt.colorbar(label='Power')
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuración ---
filename = r"C:\Users\laboratorio\Documents\GitHub\LFP-extraction-electrophysiology\LFP_output\output_file.lfp"
fs_lfp = 1250  # Frecuencia de muestreo destino
dtype = 'int16'
n_channels = 67 # Tu estimación inicial

# 1. Cargar el archivo binario
print("Cargando datos...")
data = np.fromfile(filename, dtype=dtype)
total_points = len(data)

# 2. Verificación de Canales
print(f"Total de puntos de datos leídos: {total_points}")
if total_points % n_channels != 0:
    print(f"¡ADVERTENCIA! {total_points} no es divisible exactamente entre {n_channels}.")
    
    # Buscar posibles números de canales (usualmente son múltiplos de 16, 32, 64...)
    posibles_canales = [i for i in range(1, 400) if total_points % i == 0]
    print(f"Basado en el tamaño del archivo, el número de canales podría ser uno de estos: {posibles_canales}")
    print("Por favor, corrige la variable 'n_channels' con el valor correcto de tu sistema de adquisición.")
    # Detenemos la ejecución para evitar un error en el reshape
    raise ValueError("Número de canales incorrecto.")
else:
    print(f"¡Éxito! El total de datos es divisible por {n_channels} canales.")

# 3. Reshape y creación del DataFrame de Pandas
n_samples = total_points // n_channels
data = data.reshape((n_samples, n_channels))

# Crear el DataFrame
# Nombramos las columnas CH_0, CH_1, ..., CH_N
column_names = [f'CH_{i}' for i in range(n_channels)]
df = pd.DataFrame(data, columns=column_names)

# Establecer el índice del DataFrame como el tiempo en segundos
df.index = np.arange(n_samples) / fs_lfp
df.index.name = 'Tiempo (s)'

print(f"\nResumen del DataFrame:")
print(df.info())

# 4. Graficar todos los canales en una ventana de tiempo específica
t_start, t_stop = 4.0, 6.0

# Usamos .loc de Pandas para filtrar fácilmente por el índice de tiempo
df_window = df.loc[t_start:t_stop]

# Aplicar el offset (desplazamiento vertical) para separar las líneas
offset_value = 500
# Creamos un arreglo con los offsets [0, 500, 1000, 1500...] y se lo sumamos a las columnas
offsets = np.arange(n_channels) * offset_value
df_plot = df_window + offsets

# Generar el plot
plt.figure(figsize=(15, 12)) # Ajusta el tamaño de la figura para que quepan todos
ax = plt.gca()

# Pandas grafica directamente todas las columnas
df_plot.plot(ax=ax, legend=False, colormap='viridis', linewidth=0.8)

plt.title(f"LFP - Todos los canales ({n_channels})")
plt.xlabel("Tiempo (s)")
plt.ylabel(f"Amplitud (uV + offset de {offset_value})")
plt.grid(True, alpha=0.3)
plt.xlim(t_start, t_stop)

# Ajustar los "ticks" (marcas) del eje Y para que muestren el nombre del canal en su altura correspondiente
plt.yticks(offsets, column_names, fontsize=8)

# Limpiar los márgenes para ver mejor los datos
plt.tight_layout()
plt.show()
