# p8_final
# COINing a New Sound: Compressing Audio with Implicit Neural Representations

## Usage
The main.py file is for training a single SIREN model, conducting a full sweep, or fine-tuning a previous model.

The main_stitching.py file is for stitching a 10 second sound sample together by compressing 1-second segments at a time and then stitching them together at the end. This can be done by either training each from scratch or using a warm-start method.

The testing.ipynb and util.py documents the implementation of quantization, and the general testing aswell as the metric calculations conducted in the report.

## Sound samples
During this project sounds clips have been used for testing, consisting of a Classical, Rock, Pop and Speech sample. All can be found in the soundFiles folder, along with their corresponding MP3 and Opus formats.

## Trained models
All trained models used during the report can be found in the state_dicts folder.


## Dependencies that are not directly installable via Conda or Pip
This work is using ViSQOL as a metric. Installation guide can be found here: https://github.com/google/visqol
