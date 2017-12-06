"""NIPS2017 "Time Domain Neural Audio Style Transfer" code repository
Parag K. Mital
"""
import glob
import numpy as np
from scipy.signal import hann
import librosa
import matplotlib
import matplotlib.pyplot as plt
import os


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


def rainbowgram(path,
                ax,
                peak=70.0,
                use_cqt=False,
                n_fft=1024,
                hop_length=256,
                sr=22050,
                over_sample=4,
                res_factor=0.8,
                octaves=5,
                notes_per_octave=10):
    audio = librosa.load(path, sr=sr)[0]
    if use_cqt:
        C = librosa.cqt(audio,
                        sr=sr,
                        hop_length=hop_length,
                        bins_per_octave=int(notes_per_octave * over_sample),
                        n_bins=int(octaves * notes_per_octave * over_sample),
                        filter_scale=res_factor,
                        fmin=librosa.note_to_hz('C2'))
    else:
        C = librosa.stft(
            audio,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            center=True)
    mag, phase = librosa.core.magphase(C)
    phase_angle = np.angle(phase)
    phase_unwrapped = np.unwrap(phase_angle)
    dphase = phase_unwrapped[:, 1:] - phase_unwrapped[:, :-1]
    dphase = np.concatenate([phase_unwrapped[:, 0:1], dphase], axis=1) / np.pi
    mag = (librosa.logamplitude(
        mag**2, amin=1e-13, top_db=peak, ref_power=np.max) / peak) + 1
    cdict = {
        'red': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        'green': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        'blue': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        'alpha': ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0))
    }
    my_mask = matplotlib.colors.LinearSegmentedColormap('MyMask', cdict)
    plt.register_cmap(cmap=my_mask)
    ax.matshow(dphase[::-1, :], cmap=plt.cm.rainbow)
    ax.matshow(mag[::-1, :], cmap=my_mask)


def rainbowgrams(list_of_paths,
                 saveto=None,
                 rows=2,
                 cols=4,
                 col_labels=[],
                 row_labels=[],
                 use_cqt=True,
                 figsize=(15, 20),
                 peak=70.0):
    """Build a CQT rowsXcols.
    """
    N = len(list_of_paths)
    assert N == rows * cols
    fig, axes = plt.subplots(
        rows, cols, sharex=True, sharey=True, figsize=figsize)
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.05, hspace=0.1)
    #       fig = plt.figure(figsize=(18, N * 1.25))
    for i, path in enumerate(list_of_paths):
        row = int(i / cols)
        col = i % cols
        if rows == 1 and cols == 1:
            ax = axes
        elif rows == 1:
            ax = axes[col]
        elif cols == 1:
            ax = axes[row]
        else:
            ax = axes[row, col]
        rainbowgram(path, ax, peak, use_cqt)
        ax.set_axis_bgcolor('white')
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0 and row_labels:
            ax.set_ylabel(row_labels[row])
        if row == rows - 1 and col_labels:
            ax.set_xlabel(col_labels[col])
    if saveto is not None:
        fig.savefig(filename='{}.png'.format(saveto))


def plot_rainbowgrams():
    for root in ['target', 'corpus', 'results']:
        files = glob.glob('{}/**/*.wav'.format(root), recursive=True)
        for f in files:
            fname = '{}.png'.format(f)
            if not os.path.exists(fname):
                rainbowgrams(
                    [f],
                    saveto=fname,
                    figsize=(20, 5),
                    rows=1,
                    cols=1)
                plt.close('all')
