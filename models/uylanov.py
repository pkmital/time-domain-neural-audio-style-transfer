"""NIPS2017 "Time Domain Neural Audio Style Transfer" code repository
Parag K. Mital
"""
import tensorflow as tf
import librosa
import numpy as np
import argparse
import glob
import os
import utils


def read_audio_spectum(filename, n_fft=2048, hop_length=512, sr=22050):
    x, sr = librosa.load(filename, sr=sr)
    S = librosa.stft(x, n_fft, hop_length)
    S = np.log1p(np.abs(S)).T
    return S, sr


def compute_features(content,
                     style,
                     stride=1,
                     n_layers=1,
                     n_filters=4096,
                     k_h=1,
                     k_w=11):
    n_frames = content.shape[0]
    n_samples = content.shape[1]
    content_tf = np.ascontiguousarray(content)
    style_tf = np.ascontiguousarray(style)
    g = tf.Graph()
    kernels = []
    layers = []
    content_features = []
    style_features = []
    with g.as_default(), g.device('/cpu:0'), tf.Session():
        x = tf.placeholder('float32', [None, n_samples], name="x")
        net = tf.reshape(x, [1, 1, -1, n_samples])
        for layer_i in range(n_layers):
            if layer_i == 0:
                std = np.sqrt(2) * np.sqrt(2.0 / ((n_frames + n_filters) * k_w))
                kernel = np.random.randn(k_h, k_w, n_samples, n_filters) * std
            else:
                std = np.sqrt(2) * np.sqrt(2.0 / (
                    (n_filters + n_filters) * k_w))
                kernel = np.random.randn(k_h, k_w, n_filters, n_filters) * std
            kernels.append(kernel)
            kernel_tf = tf.constant(
                kernel, name="kernel{}".format(layer_i), dtype='float32')
            conv = tf.nn.conv2d(
                net,
                kernel_tf,
                strides=[1, stride, stride, 1],
                padding="VALID",
                name="conv{}".format(layer_i))
            net = tf.nn.relu(conv)
            layers.append(net)
            content_feature = net.eval(feed_dict={x: content_tf})
            content_features.append(content_feature)
            style_feature = net.eval(feed_dict={x: style_tf})
            features = np.reshape(style_feature, (-1, n_filters))
            style_gram = np.matmul(features.T, features) / n_frames
            style_features.append(style_gram)
    return content_features, style_features, kernels


def compute_stylization(kernels,
                        n_samples,
                        n_frames,
                        content_features,
                        style_features,
                        stride=1,
                        n_layers=1,
                        alpha=1e-4,
                        learning_rate=1e-3,
                        iterations=100):
    result = None
    with tf.Graph().as_default():
        x = tf.Variable(
            np.random.randn(1, 1, n_frames, n_samples).astype(np.float32) *
            1e-3,
            name="x")
        net = x
        content_loss = 0
        style_loss = 0
        for layer_i in range(n_layers):
            kernel_tf = tf.constant(
                kernels[layer_i],
                name="kernel{}".format(layer_i),
                dtype='float32')
            conv = tf.nn.conv2d(
                net,
                kernel_tf,
                strides=[1, stride, stride, 1],
                padding="VALID",
                name="conv{}".format(layer_i))
            net = tf.nn.relu(conv)
            content_loss = content_loss + \
                alpha * 2 * tf.nn.l2_loss(net - content_features[layer_i])
            _, height, width, number = map(lambda i: i.value, net.get_shape())
            feats = tf.reshape(net, (-1, number))
            gram = tf.matmul(tf.transpose(feats), feats) / n_frames
            style_loss = style_loss + 2 * tf.nn.l2_loss(gram - style_features[
                layer_i])
        loss = content_loss + style_loss
        opt = tf.contrib.opt.ScipyOptimizerInterface(
            loss, method='L-BFGS-B', options={'maxiter': iterations})
        # Optimization
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            print('Started optimization.')
            opt.minimize(sess)
            print('Final loss:', loss.eval())
            result = x.eval()
    return result


def run(content_fname,
        style_fname,
        output_fname,
        n_fft=2048,
        hop_length=256,
        alpha=0.02,
        n_layers=1,
        n_filters=8192,
        k_w=15,
        stride=1,
        iterations=300,
        phase_iterations=500,
        sr=22050,
        signal_length=1,  # second
        block_length=1024):

    content, sr = read_audio_spectum(
        content_fname, n_fft=n_fft, hop_length=hop_length, sr=sr)
    style, sr = read_audio_spectum(
        style_fname, n_fft=n_fft, hop_length=hop_length, sr=sr)

    n_frames = min(content.shape[0], style.shape[0])
    n_samples = content.shape[1]
    content = content[:n_frames, :]
    style = style[:n_frames, :]

    content_features, style_features, kernels = compute_features(
        content=content,
        style=style,
        stride=stride,
        n_layers=n_layers,
        n_filters=n_filters,
        k_w=k_w)

    result = compute_stylization(
        kernels=kernels,
        n_samples=n_samples,
        n_frames=n_frames,
        content_features=content_features,
        style_features=style_features,
        stride=stride,
        n_layers=n_layers,
        alpha=alpha,
        iterations=iterations)

    mags = np.zeros_like(content.T)
    mags[:, :n_frames] = np.exp(result[0, 0].T) - 1

    p = 2 * np.pi * np.random.random_sample(mags.shape) - np.pi
    for i in range(phase_iterations):
        S = mags * np.exp(1j * p)
        x = librosa.istft(S, hop_length)
        p = np.angle(librosa.stft(x, n_fft, hop_length))

    librosa.output.write_wav('prelimiter.wav', x, sr)
    limited = utils.limiter(x)
    librosa.output.write_wav(output_fname, limited, sr)


def batch(content_path, style_path, output_path):
    content_files = glob.glob('{}/*.wav'.format(content_path))
    style_files = glob.glob('{}/*.wav'.format(style_path))
    for content_filename in content_files:
        for style_filename in style_files:
            output_filename = '{}/{}+{}.wav'.format(
                output_path,
                content_filename.split('/')[-1], style_filename.split('/')[-1])
            if os.path.exists(output_filename):
                continue
            run(content_filename, style_filename, output_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--style', help='style file', required=True)
    parser.add_argument('-c', '--content', help='content file', required=True)
    parser.add_argument('-o', '--output', help='output file', required=True)
    parser.add_argument(
        '-m',
        '--mode',
        help='mode for training [single] or batch',
        default='single')

    args = vars(parser.parse_args())
    if args['mode'] == 'single':
        run(args['content'], args['style'], args['output'])
    else:
        batch(args['content'], args['style'], args['output'])
