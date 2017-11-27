# make interface for each model type
# - original, dct, wavenet, nsynth
# parameters for each model type
# - original:
#     - layers = [1, 2, 3]
#     - frame size = [512, 2048]
#     - kernel size = [5, 15, 30]
# - dct:
#     - features = [[real, imag], [real, imag, mag],
#                   [mag, phase], [mag, phase diff],
#                   [mag, unwrapped phase diff]]
# - wavenet:
#
# - nsynth:
#
import os
import glob
import numpy as np
from models import fourier9


def batch(content_path, style_path, output_path):
    content_files = glob.glob('{}/*.wav'.format(content_path))
    style_files = glob.glob('{}/*.wav'.format(style_path))
    content_filename = np.random.choice(content_files)
    style_filename = np.random.choice(style_files)
    n_fft = np.random.choice([2048, 4096, 8192])
    n_layers = np.random.choice([1, 2])
    n_filters = np.random.choice([1024, 2048, 4096])
    hop_length = np.random.choice([128, 256, 512])
    alpha = np.random.choice(
        [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001])
    k_w = np.random.choice([4, 8, 16])
    output_filename = (
        '{}/fourier-9/n_fft={},n_layers={},'
        'n_filters={},hop_length={},alpha={},k_w={},{}+{}.wav'.format(
            output_path, n_fft, n_layers, n_filters, hop_length, alpha, k_w,
            content_filename.split('/')[-1], style_filename.split('/')[-1]))
    print(output_filename)
    if not os.path.exists(output_filename):
        output_dir = os.path.join(output_path, 'fourier-9')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fourier9.run(content_fname=content_filename,
                     style_fname=style_filename,
                     output_fname=output_filename,
                     n_fft=n_fft,
                     n_layers=n_layers,
                     n_filters=n_filters,
                     hop_length=hop_length,
                     alpha=alpha,
                     k_w=k_w)
    # output_filename = '{}/nsynth/{}+{}.wav'.format(
    #     output_path,
    #     content_filename.split('/')[-1], style_filename.split('/')[-1])
    # if not os.path.exists(output_filename):
    #     nsynth.run(content_filename, style_filename, output_filename)
    # output_filename = '{}/original/{}+{}.wav'.format(
    #     output_path,
    #     content_filename.split('/')[-1], style_filename.split('/')[-1])
    # if not os.path.exists(output_filename):
    #     original.run(content_filename, style_filename, output_filename)


if __name__ == '__main__':
    content_path = './target'
    style_path = './corpus'
    output_path = './results'
    batch(content_path, style_path, output_path)
