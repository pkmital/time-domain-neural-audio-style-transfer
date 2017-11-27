import numpy as np
from scipy.signal import hann


def limiter(signal,
            delay=40,
            threshold=0.9,
            release_coeff=0.9995,
            attack_coeff=0.9):

    delay_index = 0
    envelope = 0
    gain = 1
    delay = delay
    delay_line = np.zeros(delay)
    release_coeff = release_coeff
    attack_coeff = attack_coeff
    threshold = threshold

    for idx, sample in enumerate(signal):
        delay_line[delay_index] = sample
        delay_index = (delay_index + 1) % delay

        # calculate an envelope of the signal
        envelope = max(np.abs(sample), envelope * release_coeff)

        if envelope > threshold:
            target_gain = threshold / envelope
        else:
            target_gain = 1.0

        # have gain go towards a desired limiter gain
        gain = (gain * attack_coeff + target_gain * (1 - attack_coeff))

        # limit the delayed signal
        signal[idx] = delay_line[delay_index] * gain
    return signal


def chop(signal, hop_size=256, frame_size=512):
    n_hops = len(signal) // hop_size
    frames = []
    hann_win = hann(frame_size)
    for hop_i in range(n_hops):
        frame = signal[(hop_i * hop_size):(hop_i * hop_size + frame_size)]
        frame = np.pad(frame, (0, frame_size - len(frame)), 'constant')
        frame *= hann_win
        frames.append(frame)
    frames = np.array(frames)
    return frames


def unchop(frames, hop_size=256, frame_size=512):
    signal = np.zeros((frames.shape[0] * hop_size + frame_size,))
    for hop_i, frame in enumerate(frames):
        signal[(hop_i * hop_size):(hop_i * hop_size + frame_size)] += frame
    return signal


def matrix_dft(V):
    N = len(V)
    w = np.exp(-2j * np.pi / N)
    col = np.vander([w], N, True)
    W = np.vander(col.flatten(), N, True) / np.sqrt(N)
    return np.dot(W, V)


def dft_np(signal, hop_size=256, fft_size=512):
    s = chop(signal, hop_size, fft_size)
    N = s.shape[-1]
    k = np.reshape(
        np.linspace(0.0, 2 * np.pi / N * (N // 2), N // 2), [1, N // 2])
    x = np.reshape(np.linspace(0.0, N - 1, N), [N, 1])
    freqs = np.dot(x, k)
    real = np.dot(s, np.cos(freqs)) * (2.0 / N)
    imag = np.dot(s, np.sin(freqs)) * (2.0 / N)
    return real, imag


def idft_np(re, im, hop_size=256, fft_size=512):
    N = re.shape[1] * 2
    k = np.reshape(
        np.linspace(0.0, 2 * np.pi / N * (N // 2), N // 2), [N // 2, 1])
    x = np.reshape(np.linspace(0.0, N - 1, N), [1, N])
    freqs = np.dot(k, x)
    signal = np.zeros((re.shape[0] * hop_size + fft_size,))
    recon = np.dot(re, np.cos(freqs)) + np.dot(im, np.sin(freqs))
    for hop_i, frame in enumerate(recon):
        signal[(hop_i * hop_size):(hop_i * hop_size + fft_size)] += frame
    return signal
