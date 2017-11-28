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
from models import fourier, nsynth, original


def get_path(model, output_path, content_filename, style_filename):
    output_dir = os.path.join(output_path, model)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = '{}/{}/{}+{}'.format(output_path, model,
                                           content_filename.split('/')[-1],
                                           style_filename.split('/')[-1])
    return output_filename


def params():
    n_fft = [2048, 4096]
    n_layers = [1, 2]
    n_filters = [1024, 2048, 4096]
    hop_length = [128, 256, 512]
    alpha = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    k_w = [4, 8, 16]
    return locals()


def batch(content_path, style_path, output_path):
    content_files = [f for f in glob.glob('{}/*.wav'.format(content_path)) if 'female-talking' in f]
    style_files = glob.glob('{}/*.wav'.format(style_path))
    content_filename = np.random.choice(content_files)
    style_filename = np.random.choice(style_files)
    for alpha in [0.05, 0.005]:
        for n_fft in [4096]:
            for n_layers in [1]:
                for n_filters in [4096]:
                    for hop_length in [256]:
                        for k_w in [4, 8, 16]:
                            # Run Fourier Model
                            fname = get_path('fourier', output_path, content_filename, style_filename)
                            output_filename = ('{},n_fft={},n_layers={},n_filters={},'
                                               'hop_length={},alpha={},k_w={}.wav'.format(
                                                   fname, n_fft, n_layers, n_filters, hop_length, alpha, k_w))
                            print(output_filename)
                            if not os.path.exists(output_filename):
                                fourier.run(content_fname=content_filename,
                                            style_fname=style_filename,
                                            output_fname=output_filename,
                                            n_fft=n_fft,
                                            n_layers=n_layers,
                                            n_filters=n_filters,
                                            hop_length=hop_length,
                                            alpha=alpha,
                                            k_w=k_w)
                            # # Run NSynth Encoder Model
                            # output_filename = get_path('nsynth-encoder', output_path, content_filename,
                            #                            style_filename)
                            # output_filename = ('{},n_fft={},n_layers={},n_filters={},'
                            #                    'hop_length={},alpha={},k_w={}.wav'.format(
                            #                        fname, n_fft, n_layers, n_filters, hop_length, alpha, k_w))
                            # if not os.path.exists(output_filename):
                            #     nsynth.run(content_filename,
                            #                style_filename,
                            #                output_filename,
                            #                model='encoder',
                            #                n_fft=n_fft,
                            #                n_layers=n_layers,
                            #                n_filters=n_filters,
                            #                hop_length=hop_length,
                            #                alpha=alpha,
                            #                k_w=k_w)
                            # # Run NSynth Decoder Model
                            # output_filename = get_path('wavenet-decoder', output_path, content_filename,
                            #                            style_filename)
                            # output_filename = ('{},n_fft={},n_layers={},n_filters={},'
                            #                    'hop_length={},alpha={},k_w={}.wav'.format(
                            #                        fname, n_fft, n_layers, n_filters, hop_length, alpha, k_w))
                            # if not os.path.exists(output_filename):
                            #     nsynth.run(content_filename,
                            #                style_filename,
                            #                output_filename,
                            #                model='decoder',
                            #                n_fft=n_fft,
                            #                n_layers=n_layers,
                            #                n_filters=n_filters,
                            #                hop_length=hop_length,
                            #                alpha=alpha,
                            #                k_w=k_w)
                            # Run Original Model
                            fname = get_path('original', output_path, content_filename, style_filename)
                            output_filename = ('{},n_fft={},n_layers={},n_filters={},'
                                               'hop_length={},alpha={},k_w={}.wav'.format(
                                                   fname, n_fft, n_layers, n_filters, hop_length, alpha, k_w))
                            print(output_filename)
                            if not os.path.exists(output_filename):
                                original.run(content_filename,
                                             style_filename,
                                             output_filename,
                                             n_fft=n_fft,
                                             n_layers=n_layers,
                                             n_filters=n_filters,
                                             hop_length=hop_length,
                                             alpha=alpha,
                                             k_w=k_w)


if __name__ == '__main__':
    content_path = './target'
    style_path = './corpus'
    output_path = './results'
    batch(content_path, style_path, output_path)
