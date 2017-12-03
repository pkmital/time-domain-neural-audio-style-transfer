"""NIPS2017 "Time Domain Neural Audio Style Transfer" code repository
Parag K. Mital
"""
import os
import glob
import numpy as np
from models import timedomain, uylanov


def get_path(model, output_path, content_filename, style_filename):
    output_dir = os.path.join(output_path, model)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = '{}/{}/{}+{}'.format(output_path, model,
                                           content_filename.split('/')[-1],
                                           style_filename.split('/')[-1])
    return output_filename


def params():
    n_fft = [2048]
    n_layers = [1]
    n_filters = [4096]
    hop_length = [256, 512]
    alpha = [0.01]
    k_w = [8]
    norm = [True, False]
    input_features = [['mags'], ['mags', 'phase'], ['real', 'imag'],
                      ['real', 'imag', 'mags']]
    return locals()


def batch(content_path, style_path, output_path):
    content_files = glob.glob('{}/*.wav'.format(content_path))
    style_files = glob.glob('{}/*.wav'.format(style_path))
    content_filename = np.random.choice(content_files)
    style_filename = np.random.choice(style_files)
    alpha = np.random.choice(params()['alpha'])
    n_fft = np.random.choice(params()['n_fft'])
    n_layers = np.random.choice(params()['n_layers'])
    n_filters = np.random.choice(params()['n_filters'])
    hop_length = np.random.choice(params()['hop_length'])
    norm = np.random.choice(params()['norm'])
    k_w = np.random.choice(params()['k_w'])

    # Run the Time Domain Model
    for f in params()['input_features']:
        fname = get_path('timedomain/input_features={}'.format(",".join(f)),
                         output_path, content_filename, style_filename)
        output_filename = ('{},n_fft={},n_layers={},n_filters={},norm={},'
                           'hop_length={},alpha={},k_w={}.wav'.format(
                               fname, n_fft, n_layers, n_filters, norm,
                               hop_length, alpha, k_w))
        print(output_filename)
        if not os.path.exists(output_filename):
            timedomain.run(content_fname=content_filename,
                           style_fname=style_filename,
                           output_fname=output_filename,
                           n_fft=n_fft,
                           n_layers=n_layers,
                           n_filters=n_filters,
                           hop_length=hop_length,
                           alpha=alpha,
                           norm=norm,
                           k_w=k_w)

    # Run Original Uylanov Model
    fname = get_path('uylanov', output_path, content_filename, style_filename)
    output_filename = ('{},n_fft={},n_layers={},n_filters={},'
                       'hop_length={},alpha={},k_w={}.wav'.format(
                           fname, n_fft, n_layers, n_filters, hop_length, alpha,
                           k_w))
    print(output_filename)
    if not os.path.exists(output_filename):
        uylanov.run(content_filename,
                    style_filename,
                    output_filename,
                    n_fft=n_fft,
                    n_layers=n_layers,
                    n_filters=n_filters,
                    hop_length=hop_length,
                    alpha=alpha,
                    k_w=k_w)

    # These only produce noise so they are commented
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


if __name__ == '__main__':
    content_path = './target'
    style_path = './corpus'
    output_path = './results'
    batch(content_path, style_path, output_path)
