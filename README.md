# Time Domain Neural Audio Style Transfer

NIPS2017 "Time Domain Neural Audio Style Transfer" code repository  
https://arxiv.org/abs/1711.11160  
Parag K. Mital  
  
Presented at [nips2017creativity.github.io](nips2017creativity.github.io)

# Introduction

  A recently published method for audio style transfer has shown how to extend the process of image style transfer to audio.  This method synthesizes audio "content" and "style" independently using the magnitudes of a short time Fourier transform, shallow convolutional networks with randomly initialized filters, and iterative phase reconstruction with Griffin-Lim.  In this work, we explore whether it is possible to directly optimize a time domain audio signal, removing the process of phase reconstruction and opening up possibilities for real-time applications and higher quality syntheses.  We explore a variety of style transfer processes on neural networks that operate directly on time domain audio signals and demonstrate one such network capable of audio stylization.

# Installation

Python 3.4+ required (Magenta is required for NSynth and WaveNet models only; but I was unable to stylize audio using these models).

# Code

The `models` folder shows three different modules, `timedomain` shows the novel work showing how to combine the `real`, `imaginary`, and `magnitudes` of a discrete timedomain transform for performing time-domain audio style transfer.  The `uylanov` module includes the approach by Ulyanov et al.  Finally, the `nsynth` module includes the NSynth Autoencoder, showing how to use the encoder or the decoder as approaches to audio stylization, though I was unable to perform any successful stylization using this approach.

# Usage

You can use any of the modules in the models folder, `timedomain`, `original`, or `nsynth` from the command line like so:

```
python models/timedomain.py
usage: timedomain.py [-h] -s STYLE -c CONTENT -o OUTPUT [-m MODE]
```

These take paths to the style or content files or paths (when mode is 'batch'), e.g.:

```
python models/timedomain.py -s /path/to/style.wav -c /path/to/content.wav -o /path/to/output.wav
```

or:

```
python models/timedomain.py -s /path/to/style/wavs/folder -c /path/to/content/wavs/folder -o /path/to/output/wavs/folder -m batch
```

# Audio Samples

This repository also includes audio samples from Robert Thomas (`target/male-talking.wav`), music by [Robert Thomas](http://robertthomassound.com/) and [Franky Redente](https://soundcloud.com/franky80y) (`corpus/robthomas*`), samples and music by [John Tejada](http://www.paletterecordings.com/) and [Reggie Watts](http://reggiewatts.com/) (`corpus/johntejada*`, `target/beat-box*`, `target/male-singing.wav`, `target/female-singing.wav`), and one voice sample by [Ashwin Vaswani](https://www.ashwinvaswani.com/) (`target/male-taal.wav`).  These clips were generously contributed to this work by their authors and are *licensed under a Creative Commons Attribution-NonCommercial 4.0 International License*.  That means these clips are not for commercial usage.  Further, any sharing of these clips must contain attribution to their authors, and must be shared under the same license.

# Example Outputs

The folder `examples` includes syntheses using the `models/timedomain` module and the original Ulyanov network in `models/uylanov`, and were created using the script in the root of the repo, `search.py`.
`
