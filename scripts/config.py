import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
# print("tensorflow version: {}".format(tf.__version__))
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"]="0"


ROOT = "/mount/arbeitsdaten/asr-2/baofg/"
CORPUS = ROOT + "corpus/"
RESULTS = ROOT + "results/"
SRC = ROOT + 'master-thesis/src/'

sys.path.append(SRC)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
GPU_CONFIG = tf.ConfigProto(gpu_options=gpu_options)
