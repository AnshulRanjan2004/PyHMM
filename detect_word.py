"""
This module detects the word given the audio file using HMM techniques
1. We need to train 3 HMMs to detect 3 words: go, down, stop
2. Create obs for each file for all files from each directory for go, down, stop
3. Using the observations and some N states, train the corresponding model
4. Given a new file, compute the observations, pass it through the models and find argmax
"""
import json
import os
import pickle
import random
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
from codebook_creator import create_mfcc_dataset_for_codebook, collect_training_data, get_mfcc_vectors
from myhmm_scaled import MyHmmScaled

model_file_name = r"./models/a.json"
BASE_DATAPATH = "data"
STOP_PATH = "data\stop"
DOWN_PATH = "data\down"
GO_PATH = "data\go"


def create_initial_model(num_states, num_symbols, model_name=None):
    """
    create an initial lambda with pi, aij, bjk suitable for codebook size
    write the model to model_name
    :return:
    """
    pi = create_initial_distribution(num_states)
    aij = create_aij(num_states)
    bjk = create_bjk(num_states, num_symbols)
    model = {
        "A": aij,
        "B": bjk,
        "pi": pi,
    }
    model = {"hmm": model}
    if model_name is not None:
        val = json.dumps(model)
        with open(model_name, "w") as f:
            f.write(val)
    return model


def create_initial_distribution(num_states):
    pi1 = np.random.dirichlet(np.ones(num_states), size=1)[0].tolist()
    assert sum(pi1) == 1, pi1
    pi = dict()
    for i, val in enumerate(pi1):
        pi[i] = val
    return pi


def create_aij(num_states):
    aij = {}
    for i in range(num_states):
        data = np.random.dirichlet(np.ones(num_states), size=1)[0].tolist()
        aij[i] = {}
        for j, val in enumerate(data):
            aij[i][j] = val
    return aij


def create_bjk(num_states, num_symbols):
    bjk = {}
    for i in range(num_states):
        data = np.random.dirichlet(np.ones(num_symbols), size=1)[0].tolist()
        bjk[i] = {}
        for j, val in enumerate(data):
            bjk[i][j] = val
    return bjk


def get_vecs(label, num_files=100):
    vecs_list = []
    file_names = collect_training_data(label, num_files=num_files)
    for name in file_names:
        vecs = get_mfcc_vectors([name])
        vecs_list.append(vecs)
    return vecs_list


def get_obs(vecs, book):
    codes = vq(np.array(vecs), book)[0]
    codes = [str(code) for code in codes]
    return codes


def get_obs_list(vecs_list):
    obs_list = []
    for vecs in vecs_list:
        obs = get_obs(vecs, book)
        obs_list.append(obs)
    return obs_list


def train(hmm, obs_list):
    hmm.forward_backward_multi_scaled(obs_list)
    return hmm


def classify(models, obs):
    probs = {}
    for k, v in models.items():
        prob = v.forward_scaled(obs)
        probs[k] = prob
    print("probs: ", probs)
    keys = list(probs.keys())
    vals = list(probs.values())
    val = max(vals)
    index = vals.index(val)
    key = keys[index]
    # print("Predicted Class = ", key)
    return key


if __name__ == '__main__':
    # v = create_initial_model(2, 64, "./models/a.json")
    # print(v)
    labels = ["go", "stop",]
    labels = ["stop", "go"]

    book = pickle.load(open("book.p", "rb"))

    models = {
    }

    for label in labels:
        hmm = MyHmmScaled(model_file_name)
        vecs_list = get_vecs(label, 200)
        obs_list = get_obs_list(vecs_list)
        print(label)
        # print(len(obs_list), len(obs_list[0]))
        # print(obs_list[33])
        hmm = train(hmm, obs_list)
        models[label] = hmm

    pickle.dump(models, open("models.p", "wb"))
    print(models)
    classify(models, obs_list[12])