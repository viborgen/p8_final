# p8_final
# COINing a New Sound: Compressing Audio with Implicit Neural Representations

## Sound samples
During this project sounds clips have been used for testing, consisting of a Classical, Rock, Pop and Speech sample. All can be found in the soundFiles folder, along with their corresponding MP3 and Opus variants.

## Useage
the main.py file is for running a single test run, og conducting a full sweep. Booleans for turning sweeping or retraining off and on can be found.
the main_stitching.py file is for stitching a 10 second sound sample together by compressing 1 second at a time and then stitching it together at the end. This can be done by either training each from scratch or using a warm start method. Weights & Biases can be enabled in a boolean aswell.



## dependencies that cannot be installed with conda or pip
This work is using ViSQOL as a metric. Installation guide can be found here: https://github.com/google/visqol
