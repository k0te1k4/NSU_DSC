import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


# Function for the rectangular signal
def rectangular_signal(t, big_t):
    A = 2
    return np.where(np.mod(t, big_t) > big_t / 2, A, -A)


# Function for the cosine signal
def cosine_func(t, A, w):
    return A * np.cos(w * t)


# Fourier coefficients calculation for the signal
def an(t0, func, big_t, n, w):
    return (2 / big_t) * integrate.quad(lambda t: func(t) * np.cos(n * w * t), t0, t0 + big_t)[0]


def bn(t0, func, big_t, n, w):
    return (2 / big_t) * integrate.quad(lambda t: func(t) * np.sin(n * w * t), t0, t0 + big_t)[0]


def approx_func(t0, func, t, big_t, w, big_n):
    a0 = an(t0, func, big_t, 0, w)
    summ = a0 / 2
    for n in range(1, big_n):
        summ += an(t0, func, big_t, n, w) * np.cos(n * w * t) + bn(t0, func, big_t, n, w) * np.sin(n * w * t)
    return summ


# Compute FFT and return the spectrum and frequencies
def compute_fft(signal, t):
    signal_fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(t), t[1] - t[0])
    return signal_fft, freq


if __name__ == '__main__':
    # Parameters
    A = 2  # Amplitude
    f = 100  # Frequency
    w = 2 * np.pi * f  # Angular frequency
    big_t = 1 / f  # Period
    big_n = 10
    t0 = 0

    # Time vector for plotting signals (small interval around 0)
    t = np.arange(-0.02, 0.02, 0.0001)

    # Generate the cosine signal
    cos_signal = cosine_func(t, A, w)

    # === Опорный график: прямоугольный сигнал и его аппроксимация ===

    plot_t = np.arange(-4, 4, 0.01)
    rect_fun = list(map(lambda a: rectangular_signal(a, 2), plot_t))
    rect_approx = list(map(lambda a: approx_func(0, lambda t: rectangular_signal(t, 2), a, 2, np.pi, 10), plot_t))
    rect_err = np.subtract(np.array(rect_approx), np.array(rect_fun))

    # График сигнала прямоугольной формы и его аппроксимации
    plt.figure()
    plt.plot(plot_t, rect_fun, label='Rectangular Signal')
    plt.plot(plot_t, rect_approx, label='Approximation', linestyle='dashed')
    plt.legend()
    plt.grid()
    plt.show()

    # График ошибки аппроксимации
    plt.figure()
    plt.plot(plot_t, rect_err, label='Approximation Error', color='red')
    plt.title('Approximation Error')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.grid()
    plt.show()

    # === Cosine Signal and Spectrum ===

    # Plot the cosine signal
    plt.figure()
    plt.plot(t, cos_signal, label='Cosine Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Cosine Signal x(t) = A*cos(wt)')
    plt.grid()
    plt.show()

    # Fourier Transform of the cosine signal
    signal_fft, freq = compute_fft(cos_signal, t)

    # Plot the spectrum (Magnitude of FFT)
    plt.figure()
    plt.plot(np.fft.fftshift(freq), np.fft.fftshift(np.abs(signal_fft)), label="Cosine Signal Spectrum")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Fourier Spectrum of Cosine Signal')
    plt.grid()
    plt.show()

    # Compute Fourier coefficients a_n for the cosine signal
    coeffs = [an(t0, lambda t: cosine_func(t, A, w), big_t, n, w) for n in range(big_n)]

    # Plot the spectrum with Fourier coefficients marked
    plt.figure()
    plt.plot(np.fft.fftshift(freq), np.fft.fftshift(np.abs(signal_fft)), label="Cosine Signal Spectrum")
    plt.vlines(f, 0, max(np.abs(signal_fft)), colors='orange', linestyles='dashed', label='a_n (Fourier Coefficient)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Fourier Spectrum of Cosine Signal with a_n')
    plt.grid()
    plt.legend()
    plt.show()

    # === Rectangular Signal and Spectrum ===

    # Generate the rectangular signal
    rect_signal = rectangular_signal(t, big_t)

    # Plot the rectangular signal
    plt.figure()
    plt.plot(t, rect_signal, label='Rectangular Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Rectangular Signal')
    plt.grid()
    plt.show()

    # FFT of the rectangular signal
    rect_fft, rect_freq = compute_fft(rect_signal, t)

    # Plot the rectangular signal spectrum
    plt.figure()
    plt.plot(np.fft.fftshift(rect_freq), np.fft.fftshift(np.abs(rect_fft)), label='Rectangular Signal Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Fourier Spectrum of Rectangular Signal')
    plt.grid()
    plt.show()

    # === Noisy Rectangular Signal and Spectrum ===

    # Adding noise to the rectangular signal
    noise = np.random.normal(0, 0.5, len(t))  # Gaussian noise with mean=0 and std=0.5
    noisy_signal = rect_signal + noise

    # Plot the noisy signal
    plt.figure()
    plt.plot(t, noisy_signal, label='Noisy Rectangular Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Noisy Rectangular Signal')
    plt.grid()
    plt.show()

    # FFT of the noisy signal
    noisy_fft, noisy_freq = compute_fft(noisy_signal, t)

    # Plot the spectrum of the noisy signal
    plt.figure()
    plt.plot(np.fft.fftshift(noisy_freq), np.fft.fftshift(np.abs(noisy_fft)), label='Noisy Signal Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Fourier Spectrum of Noisy Signal')
    plt.grid()
    plt.show()
