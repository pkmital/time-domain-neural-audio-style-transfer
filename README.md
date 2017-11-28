# neural-audio-style-transfer

NIPS2017 "Neural Audio Style Transfer" code repository

Parag K. Mital

# Introduction

  A [recently published method](https://github.com/DmitryUlyanov/neural-style-audio-tf) for audio style transfer has shown how to extend the process of image style transfer to audio.  This method synthesizes audio "content" and "style" independently using the magnitudes of a short time Fourier transform, shallow convolutional networks with randomly initialized filters, and iterative phase reconstruction with Griffin-Lim.  In this work, we explore whether it is possible to incorporate phase information into the stylization process, removing the need for phase reconstruction and opening up possibilities for real-time applications.  We build a variety of style transfer processes on neural networks including ones that incorporate phase information and ones that have been pretrained directly on time domain audio signals and show that it is possible to perform audio stylization without phase reconstruction by using the real, imaginary, and magnitude components of a Discrete Fourier Transform.

# Installation

Python 3.4+ required (Magenta is required for NSynth and WaveNet models only; but I was unable to stylize audio using these models).

# Code

The `models` folder shows three different modules, `fourier` shows the novel work showing how to combine the `real`, `imaginary`, and `magnitudes` of a discrete fourier transform for performing time-domain audio style transfer.  The `original` module includes the approach by Ulyanov.  Finally, the `nsynth` module includes the NSynth Autoencoder, showing how to use the encoder or the decoder as approaches to audio stylization, though I was unable to perform any successful stylization using this approach.

# Usage

You can use any of the modules in the models folder, `fourier`, `original`, or `nsynth` from the command line like so:

```
python models/fourier.py
usage: fourier.py [-h] -s STYLE -c CONTENT -o OUTPUT [-m MODE]
```

These take paths to the style or content files or paths (when mode is 'batch'), e.g.:

```
python models/fourier.py -s /path/to/style.wav -c /path/to/content.wav -o /path/to/output.wav
```

or:

```
python models/fourier.py -s /path/to/style/wavs/folder -c /path/to/content/wavs/folder -o /path/to/output/wavs/folder -m batch
```

# Audio Samples

This repository also includes audio samples from Robert Thomas (`target/male-talking.wav`), music by Robert Thomas and Franky Redente (`corpus/robthomas*`), samples and music by John Tejada and Reggie Watts (`corpus/johntejada*`, `target/beat-box*`, `target/male-singing.wav`, `target/female-singing.wav`), and a sample by Ashwin Vaswani (`target/male-taal.wav`).  These clips were generously contributed to this work by their authors and are *licensed under a Creative Commons Attribution-NonCommercial 4.0 International License*.  That means these clips are not for commercial usage.  Further, any sharing of these clips must contain attribution to their authors, and must be shared under the same license.
