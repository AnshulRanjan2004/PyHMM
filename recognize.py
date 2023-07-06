"""
recognizes the word given the audio file
"""
import os
import random
import pickle
from codebook_creator import BASE_DATAPATH, GO_PATH, STOP_PATH, collect_training_data, get_mfcc_vectors, get_codebook
from detect_word import get_obs, classify

models_file = "./models.p"

models = pickle.load(open(models_file, "rb"))
print(models)


def recognize():
    return


if __name__ == '__main__':
    num_files = 500
    go_files = collect_training_data("go", num_files)
    stop_files = collect_training_data("stop", num_files)
    book = pickle.load(open("book.p", "rb"))

    test = {
        "go": go_files,
        "stop": stop_files,
    }
    while True:
        label = input("Enter label (go, stop, q): ")
        if label == "q":
            break
        index = random.randint(0, 649)
        while True:
            try:
                test_file = test[label][index]
                break
            except IndexError:
                index = random.randint(0, 649)
                continue
        print("Testing for ", test_file, ", Expected label: ", label)
        vecs = get_mfcc_vectors([test_file])
        obs = get_obs(vecs, book)
        # print(obs)
        try:
            pred = classify(models, obs)
            print("Expected Label: ", label, ", Predicted: ", pred)
        except:
            print("Some exception occured due to div by zero, skipping")

