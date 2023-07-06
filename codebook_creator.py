"""
This module detects the word given the audio file using HMM techniques
"""
import os
import pickle
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
from scipy.cluster.vq import vq, kmeans, whiten

BASE_DATAPATH = "data"
STOP_PATH = "data\stop"
DOWN_PATH = "data\down"
GO_PATH = "data\go"


def collect_training_data(label, num_files):
    paths = {
        "go": GO_PATH,
        "down": DOWN_PATH,
        "stop": STOP_PATH,
    }
    mypath = paths[label]
    myfiles = [os.path.join(mypath, name) for name in os.listdir(mypath)]
    return myfiles[:num_files]


def get_mfcc_vectors(myfiles):
    vecs = []
    for myfile in myfiles:
        frequency_sampling, audio_signal = wavfile.read(myfile)
        # print(frequency_sampling, audio_signal)
        features_mfcc = mfcc(audio_signal, frequency_sampling)  # n x 13
        vecs.extend(features_mfcc)
    vecs = np.array(vecs)
    return vecs


def create_mfcc_dataset_for_codebook(num_files=1000):
    labels = ["go", "down", "stop"]
    vecs = []
    for label in labels:
        myfiles = collect_training_data(label, num_files)
        vecs1 = get_mfcc_vectors(myfiles)
        vecs.extend(vecs1)
    return vecs


def get_codebook(vecs, size=64):
    whitened = whiten(np.array(vecs))
    codebook, distortion = kmeans(whitened, size)
    return codebook


if __name__ == '__main__':
    vecs = create_mfcc_dataset_for_codebook()
    book = get_codebook(vecs)
    print(book.shape)
    pickle.dump(book, open("book.p", "wb"))

    # book = pickle.load(open("book.p", "rb"))
    # codes = vq(np.array(vecs), book)[0]
    # print(codes[200:400])

