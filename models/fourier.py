import tensorflow as tf
import librosa
import numpy as np
from scipy.signal import hann
import utils
import argparse
import glob
import os


def chop(signal, hop_size=256, frame_size=512):
    n_hops = len(signal) // hop_size
    s = []
    hann_win = hann(frame_size)
    for hop_i in range(n_hops):
        frame = signal[(hop_i * hop_size):(hop_i * hop_size + frame_size)]
        frame = np.pad(frame, (0, frame_size - len(frame)), 'constant')
        frame *= hann_win
        s.append(frame)
    s = np.array(s)
    return s


def unchop(frames, hop_size=256, frame_size=512):
    signal = np.zeros((frames.shape[0] * hop_size + frame_size,))
    for hop_i, frame in enumerate(frames):
        signal[(hop_i * hop_size):(hop_i * hop_size + frame_size)] += frame
    return signal


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


def compute_features(content,
                     style,
                     stride=1,
                     n_layers=1,
                     n_filters=4096,
                     n_fft=1024,
                     k_h=1,
                     k_w=11):
    n_frames = content.shape[0]
    n_samples = content.shape[1]
    content_tf = np.ascontiguousarray(content)
    style_tf = np.ascontiguousarray(style)
    g = tf.Graph()
    kernels = []
    content_features = []
    style_features = []
    with g.as_default(), g.device('/cpu:0'), tf.Session():
        x = tf.placeholder('float32', [n_frames, n_samples], name="x")
        p = np.reshape(
            np.linspace(0.0, n_samples - 1, n_samples), [n_samples, 1])
        k = np.reshape(
            np.linspace(0.0, 2 * np.pi / n_fft * (n_fft // 2), n_fft // 2),
            [1, n_fft // 2])
        freqs = np.dot(p, k)
        freqs_tf = tf.constant(freqs, name="freqs", dtype='float32')
        real = tf.reshape(
            tf.matmul(x, tf.cos(freqs_tf)), [1, 1, n_frames, n_fft // 2])
        imag = tf.reshape(
            tf.matmul(x, tf.sin(freqs_tf)), [1, 1, n_frames, n_fft // 2])
        mags = tf.reshape(
            tf.sqrt(tf.maximum(1e-15, real * real + imag * imag)),
            [1, 1, n_frames, n_fft // 2])
        net = tf.concat([real, mags, imag], 1)
        content_feature = net.eval(feed_dict={x: content_tf})
        content_features.append(content_feature)
        style_feature = mags.eval(feed_dict={x: style_tf})
        features = np.reshape(style_feature, (-1, n_fft // 2))
        style_gram = np.matmul(features.T, features) / (n_frames)
        style_features.append(style_gram)
        for layer_i in range(n_layers):
            if layer_i == 0:
                std = np.sqrt(2) * np.sqrt(2.0 / (
                    (n_fft / 2 + n_filters) * k_w))
                kernel = np.random.randn(k_h, k_w, n_fft // 2, n_filters) * std
            else:
                std = np.sqrt(2) * np.sqrt(2.0 / (
                    (n_filters + n_filters) * k_w))
                kernel = np.random.randn(1, k_w, n_filters, n_filters) * std
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
            content_feature = net.eval(feed_dict={x: content_tf})
            content_features.append(content_feature)
            style_feature = net.eval(feed_dict={x: style_tf})
            features = np.reshape(style_feature, (-1, n_filters))
            style_gram = np.matmul(features.T, features) / (n_frames)
            style_features.append(style_gram)
    return content_features, style_features, kernels, freqs


def compute_stylization(kernels,
                        n_samples,
                        n_frames,
                        content_features,
                        style_gram,
                        freqs,
                        stride=1,
                        n_layers=1,
                        n_fft=1024,
                        alpha=1e-4,
                        learning_rate=1e-3,
                        iterations=100,
                        optimizer='bfgs'):
    result = None
    with tf.Graph().as_default():
        x = tf.Variable(
            np.random.randn(n_frames, n_samples).astype(np.float32) * 1e-3,
            name="x")
        freqs_tf = tf.constant(freqs, name="freqs", dtype='float32')
        real = tf.reshape(
            tf.matmul(x, tf.cos(freqs_tf)), [1, 1, n_frames, n_fft // 2])
        imag = tf.reshape(
            tf.matmul(x, tf.sin(freqs_tf)), [1, 1, n_frames, n_fft // 2])
        mags = tf.reshape(
            tf.sqrt(tf.maximum(1e-15, real * real + imag * imag)),
            [1, 1, n_frames, n_fft // 2])
        net = tf.concat([real, mags, imag], 1)
        content_loss = alpha * 2 * tf.nn.l2_loss(net - content_features[0])
        feats = tf.reshape(mags, (-1, n_fft // 2))
        gram = tf.matmul(tf.transpose(feats), feats) / (n_frames)
        style_loss = 2 * tf.nn.l2_loss(gram - style_gram[0])
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
                alpha * 2 * tf.nn.l2_loss(net - content_features[layer_i + 1])
            _, height, width, number = map(lambda i: i.value, net.get_shape())
            feats = tf.reshape(net, (-1, number))
            gram = tf.matmul(tf.transpose(feats), feats) / (n_frames)
            style_loss = style_loss + 2 * tf.nn.l2_loss(gram - style_gram[
                layer_i + 1])
        loss = content_loss + style_loss
        if optimizer == 'bfgs':
            opt = tf.contrib.opt.ScipyOptimizerInterface(
                loss, method='L-BFGS-B', options={'maxiter': iterations})
            # Optimization
            with tf.Session() as sess:
                sess.run(tf.initialize_all_variables())
                print('Started optimization.')
                opt.minimize(sess)
                result = x.eval()
        else:
            opt = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(loss)
            # Optimization
            with tf.Session() as sess:
                sess.run(tf.initialize_all_variables())
                print('Started optimization.')
                for i in range(iterations):
                    s, c, l, _ = sess.run([style_loss, content_loss, loss, opt])
                    print('Style:', s, 'Content:', c, end='\r')
                result = x.eval()
    return result


def run(content_fname,
        style_fname,
        output_fname,
        n_fft=4096,
        n_layers=1,
        n_filters=4096,
        hop_length=256,
        alpha=0.05,
        k_w=15,
        k_h=3,
        optimizer='bfgs',
        stride=1,
        iterations=300,
        sr=22050):

    frame_size = n_fft // 2

    audio, fs = librosa.load(content_fname, sr=sr)
    content = chop(audio, hop_size=hop_length, frame_size=frame_size)
    audio, fs = librosa.load(style_fname, sr=sr)
    style = chop(audio, hop_size=hop_length, frame_size=frame_size)

    n_frames = min(content.shape[0], style.shape[0])
    n_samples = min(content.shape[1], style.shape[1])
    content = content[:n_frames, :n_samples]
    style = style[:n_frames, :n_samples]

    content_features, style_gram, kernels, freqs = compute_features(
        content=content,
        style=style,
        stride=stride,
        n_fft=n_fft,
        n_layers=n_layers,
        n_filters=n_filters,
        k_w=k_w,
        k_h=k_h)

    result = compute_stylization(
        kernels=kernels,
        freqs=freqs,
        n_samples=n_samples,
        n_frames=n_frames,
        n_fft=n_fft,
        content_features=content_features,
        style_gram=style_gram,
        stride=stride,
        n_layers=n_layers,
        alpha=alpha,
        optimizer=optimizer,
        iterations=iterations)

    s = unchop(result, hop_size=hop_length, frame_size=frame_size)
    librosa.output.write_wav(output_fname, s, sr=sr)
    s = utils.limiter(s)
    librosa.output.write_wav(output_fname + '.limiter.wav', s, sr=sr)


def batch(content_path, style_path, output_path, model):
    content_files = glob.glob('{}/*.wav'.format(content_path))
    style_files = glob.glob('{}/*.wav'.format(style_path))
    for content_fname in content_files:
        for style_fname in style_files:
            output_fname = '{}/{}+{}.wav'.format(output_path,
                                                 content_fname.split('/')[-1],
                                                 style_fname.split('/')[-1])
            if os.path.exists(output_fname):
                continue
            run(content_fname, style_fname, output_fname, model)


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
