"""NSynth & WaveNet Audio Style Transfer."""
import os
import glob
import librosa
import argparse
import numpy as np
import tensorflow as tf
from magenta.models.nsynth.wavenet import masked
from magenta.models.nsynth.utils import mu_law, inv_mu_law_numpy
import utils


def compute_wavenet_encoder_features(content, style):
    ae_hop_length = 512
    ae_bottleneck_width = 16
    ae_num_stages = 10
    ae_num_layers = 30
    ae_filter_length = 3
    ae_width = 128
    # Encode the source with 8-bit Mu-Law.
    n_frames = content.shape[0]
    n_samples = content.shape[1]
    content_tf = np.ascontiguousarray(content)
    style_tf = np.ascontiguousarray(style)
    g = tf.Graph()
    content_features = []
    style_features = []
    layers = []
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        x = tf.placeholder('float32', [n_frames, n_samples], name="x")
        x_quantized = mu_law(x)
        x_scaled = tf.cast(x_quantized, tf.float32) / 128.0
        x_scaled = tf.expand_dims(x_scaled, 2)
        en = masked.conv1d(
            x_scaled,
            causal=False,
            num_filters=ae_width,
            filter_length=ae_filter_length,
            name='ae_startconv')
        for num_layer in range(ae_num_layers):
            dilation = 2**(num_layer % ae_num_stages)
            d = tf.nn.relu(en)
            d = masked.conv1d(
                d,
                causal=False,
                num_filters=ae_width,
                filter_length=ae_filter_length,
                dilation=dilation,
                name='ae_dilatedconv_%d' % (num_layer + 1))
            d = tf.nn.relu(d)
            en += masked.conv1d(
                d,
                num_filters=ae_width,
                filter_length=1,
                name='ae_res_%d' % (num_layer + 1))
            layers.append(en)
        en = masked.conv1d(
            en,
            num_filters=ae_bottleneck_width,
            filter_length=1,
            name='ae_bottleneck')
        en = masked.pool1d(en, ae_hop_length, name='ae_pool', mode='avg')
        saver = tf.train.Saver()
        saver.restore(sess, './model.ckpt-200000')
        content_features = sess.run(layers, feed_dict={x: content_tf})
        styles = sess.run(layers, feed_dict={x: style_tf})
        for i, style_feature in enumerate(styles):
            n_features = np.prod(layers[i].shape.as_list()[-1])
            features = np.reshape(style_feature, (-1, n_features))
            style_gram = np.matmul(features.T, features) / (n_samples *
                                                            n_frames)
            style_features.append(style_gram)
    return content_features, style_features


def compute_wavenet_encoder_stylization(n_samples,
                                        n_frames,
                                        content_features,
                                        style_features,
                                        alpha=1e-4,
                                        learning_rate=1e-3,
                                        iterations=100):
    ae_style_layers = [1, 5]
    ae_num_layers = 30
    ae_num_stages = 10
    ae_filter_length = 3
    ae_width = 128
    layers = []
    with tf.Graph().as_default() as g, g.device('/cpu:0'), tf.Session() as sess:
        x = tf.placeholder(
            name="x", shape=(n_frames, n_samples, 1), dtype=tf.float32)
        en = masked.conv1d(
            x,
            causal=False,
            num_filters=ae_width,
            filter_length=ae_filter_length,
            name='ae_startconv')
        for num_layer in range(ae_num_layers):
            dilation = 2**(num_layer % ae_num_stages)
            d = tf.nn.relu(en)
            d = masked.conv1d(
                d,
                causal=False,
                num_filters=ae_width,
                filter_length=ae_filter_length,
                dilation=dilation,
                name='ae_dilatedconv_%d' % (num_layer + 1))
            d = tf.nn.relu(d)
            en += masked.conv1d(
                d,
                num_filters=ae_width,
                filter_length=1,
                name='ae_res_%d' % (num_layer + 1))
            layer_i = tf.identity(en, name='layer_{}'.format(num_layer))
            layers.append(layer_i)
        saver = tf.train.Saver()
        saver.restore(sess, './model.ckpt-200000')
        sess.run(tf.initialize_all_variables())
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, [en.name.replace(':0', '')] +
            ['layer_{}'.format(i) for i in range(ae_num_layers)])
    with tf.Graph().as_default() as g, g.device('/cpu:0'), tf.Session() as sess:
        x = tf.Variable(
            np.random.randn(n_frames, n_samples, 1).astype(np.float32))
        tf.import_graph_def(frozen_graph_def, input_map={'x:0': x})
        content_loss = np.float32(0.0)
        style_loss = np.float32(0.0)
        for num_layer in ae_style_layers:
            layer_i = g.get_tensor_by_name(name='import/layer_%d:0' %
                                           (num_layer))
            content_loss = content_loss + alpha * 2 * tf.nn.l2_loss(
                layer_i - content_features[num_layer])
            n_features = layer_i.shape.as_list()[-1]
            features = tf.reshape(layer_i, (-1, n_features))
            gram = tf.matmul(tf.transpose(features), features) / (n_frames *
                                                                  n_samples)
            style_loss = style_loss + 2 * tf.nn.l2_loss(gram - style_features[
                num_layer])
        loss = content_loss + style_loss
        # Optimization
        print('Started optimization.')
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        var_list = tf.trainable_variables()
        print(var_list)
        sess.run(tf.initialize_all_variables())
        for i in range(iterations):
            s, c, layer, _ = sess.run([style_loss, content_loss, loss, opt])
            print(i, '- Style:', s, 'Content:', c, end='\r')
        result = x.eval()
        result = inv_mu_law_numpy(result[..., 0] / result.max() * 128.0)
    return result


def compute_wavenet_decoder_features(content, style):
    num_stages = 10
    num_layers = 30
    filter_length = 3
    width = 512
    skip_width = 256
    # Encode the source with 8-bit Mu-Law.
    n_frames = content.shape[0]
    n_samples = content.shape[1]
    content_tf = np.ascontiguousarray(content)
    style_tf = np.ascontiguousarray(style)
    g = tf.Graph()
    content_features = []
    style_features = []
    layers = []
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        x = tf.placeholder('float32', [n_frames, n_samples], name="x")
        x_quantized = mu_law(x)
        x_scaled = tf.cast(x_quantized, tf.float32) / 128.0
        x_scaled = tf.expand_dims(x_scaled, 2)
        layer = x_scaled
        layer = masked.conv1d(
            layer, num_filters=width, filter_length=filter_length, name='startconv')

        # Set up skip connections.
        s = masked.conv1d(
            layer, num_filters=skip_width, filter_length=1, name='skip_start')

        # Residual blocks with skip connections.
        for i in range(num_layers):
            dilation = 2**(i % num_stages)
            d = masked.conv1d(
                layer,
                num_filters=2 * width,
                filter_length=filter_length,
                dilation=dilation,
                name='dilatedconv_%d' % (i + 1))
            assert d.get_shape().as_list()[2] % 2 == 0
            m = d.get_shape().as_list()[2] // 2
            d_sigmoid = tf.sigmoid(d[:, :, :m])
            d_tanh = tf.tanh(d[:, :, m:])
            d = d_sigmoid * d_tanh

            layer += masked.conv1d(
                d, num_filters=width, filter_length=1, name='res_%d' % (i + 1))
            s += masked.conv1d(
                d,
                num_filters=skip_width,
                filter_length=1,
                name='skip_%d' % (i + 1))
            layers.append(s)

        saver = tf.train.Saver()
        saver.restore(sess, './model.ckpt-200000')
        content_features = sess.run(layers, feed_dict={x: content_tf})
        styles = sess.run(layers, feed_dict={x: style_tf})
        for i, style_feature in enumerate(styles):
            n_features = np.prod(layers[i].shape.as_list()[-1])
            features = np.reshape(style_feature, (-1, n_features))
            style_gram = np.matmul(features.T, features) / (n_samples *
                                                            n_frames)
            style_features.append(style_gram)
    return content_features, style_features


def compute_wavenet_decoder_stylization(n_samples,
                                        n_frames,
                                        content_features,
                                        style_features,
                                        alpha=1e-4,
                                        learning_rate=1e-3,
                                        iterations=100):

    style_layers = [1, 5]
    num_stages = 10
    num_layers = 30
    filter_length = 3
    width = 512
    skip_width = 256
    layers = []
    with tf.Graph().as_default() as g, g.device('/cpu:0'), tf.Session() as sess:
        x = tf.placeholder(
            name="x", shape=(n_frames, n_samples, 1), dtype=tf.float32)
        layer = x
        layer = masked.conv1d(
            layer, num_filters=width, filter_length=filter_length, name='startconv')

        # Set up skip connections.
        s = masked.conv1d(
            layer, num_filters=skip_width, filter_length=1, name='skip_start')

        # Residual blocks with skip connections.
        for i in range(num_layers):
            dilation = 2**(i % num_stages)
            d = masked.conv1d(
                layer,
                num_filters=2 * width,
                filter_length=filter_length,
                dilation=dilation,
                name='dilatedconv_%d' % (i + 1))
            assert d.get_shape().as_list()[2] % 2 == 0
            m = d.get_shape().as_list()[2] // 2
            d_sigmoid = tf.sigmoid(d[:, :, :m])
            d_tanh = tf.tanh(d[:, :, m:])
            d = d_sigmoid * d_tanh

            layer += masked.conv1d(
                d, num_filters=width, filter_length=1, name='res_%d' % (i + 1))
            s += masked.conv1d(
                d,
                num_filters=skip_width,
                filter_length=1,
                name='skip_%d' % (i + 1))
            layer_i = tf.identity(s, name='layer_{}'.format(num_layers))
            layers.append(layer_i)
        saver = tf.train.Saver()
        saver.restore(sess, './model.ckpt-200000')
        sess.run(tf.initialize_all_variables())
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, [s.name.replace(':0', '')] +
            ['layer_{}'.format(i) for i in range(num_layers)])

    with tf.Graph().as_default() as g, g.device('/cpu:0'), tf.Session() as sess:
        x = tf.Variable(
            np.random.randn(n_frames, n_samples, 1).astype(np.float32))
        tf.import_graph_def(frozen_graph_def, input_map={'x:0': x})
        content_loss = np.float32(0.0)
        style_loss = np.float32(0.0)
        for num_layer in style_layers:
            layer_i = g.get_tensor_by_name(name='import/layer_%d:0' %
                                           (num_layer))
            content_loss = content_loss + alpha * 2 * tf.nn.l2_loss(
                layer_i - content_features[num_layer])
            n_features = layer_i.shape.as_list()[-1]
            features = tf.reshape(layer_i, (-1, n_features))
            gram = tf.matmul(tf.transpose(features), features) / (n_frames *
                                                                  n_samples)
            style_loss = style_loss + 2 * tf.nn.l2_loss(gram - style_features[
                num_layer])
        loss = content_loss + style_loss
        # Optimization
        print('Started optimization.')
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        var_list = tf.trainable_variables()
        print(var_list)
        sess.run(tf.initialize_all_variables())
        for i in range(iterations):
            s, c, _ = sess.run([style_loss, content_loss, opt])
            print(i, '- Style:', s, 'Content:', c, end='\r')
        result = x.eval()
        result = inv_mu_law_numpy(result[..., 0] / result.max() * 128.0)

    return result


def run(content_fname,
        style_fname,
        output_path,
        model,
        iterations=100,
        sr=16000,
        hop_size=512,
        frame_size=2048,
        alpha=1e-3):

    content, fs = librosa.load(content_fname, sr=sr)
    style, fs = librosa.load(style_fname, sr=sr)
    n_samples = (min(content.shape[0], style.shape[0]) // 512) * 512
    content = utils.chop(content[:n_samples], hop_size, frame_size)
    style = utils.chop(style[:n_samples], hop_size, frame_size)

    if model == 'encoder':
        content_features, style_features = compute_wavenet_encoder_features(
            content=content, style=style)
        result = compute_wavenet_encoder_stylization(
            n_frames=content_features[0].shape[0],
            n_samples=frame_size,
            alpha=alpha,
            content_features=content_features,
            style_features=style_features,
            iterations=iterations)
    elif model == 'decoder':
        content_features, style_features = compute_wavenet_decoder_features(
            content=content, style=style)
        result = compute_wavenet_decoder_stylization(
            n_frames=content_features[0].shape[0],
            n_samples=frame_size,
            alpha=alpha,
            content_features=content_features,
            style_features=style_features,
            iterations=iterations)
    else:
        raise ValueError('Unsupported model type: {}.'.format(model))

    x = utils.unchop(result, hop_size, frame_size)
    librosa.output.write_wav('prelimiter.wav', x, sr)

    limited = utils.limiter(x)
    output_fname = '{}/{}+{}.wav'.format(output_path,
                                         content_fname.split('/')[-1],
                                         style_fname.split('/')[-1])
    librosa.output.write_wav(output_fname, limited, sr=sr)


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
    parser.add_argument(
        '-s', '--style', help='style file(s) location', required=True)
    parser.add_argument(
        '-c', '--content', help='content file(s) location', required=True)
    parser.add_argument('-o', '--output', help='output path', required=True)
    parser.add_argument(
        '-m',
        '--model',
        help='model type: [encoder], or decoder',
        default='encoder')
    parser.add_argument(
        '-t',
        '--type',
        help='mode for training [single] (point to files) or batch (point to path)',
        default='single')

    args = vars(parser.parse_args())
    if args['model'] == 'single':
        run(args['content'], args['style'], args['output'], args['model'])
    else:
        batch(args['content'], args['style'], args['output'], args['model'])
