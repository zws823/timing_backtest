import numpy as np


def fourier_filter(signal, ratio):
    signal_fft = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), 1 / len(signal))

    # 设计低通滤波器
    cutoff_frequency = np.floor(ratio * len(signal))  # 截止频率
    low_pass_filter = np.abs(frequencies) < cutoff_frequency

    # 应用滤波器
    filtered_signal_fft = signal_fft * low_pass_filter

    # 计算逆傅里叶变换
    filtered_signal = np.fft.ifft(filtered_signal_fft)

    return abs(filtered_signal)
