"""
To calculate the accurarcy of the model
"""
import os
import random
import pickle
from codebook_creator import BASE_DATAPATH, GO_PATH, STOP_PATH, collect_training_data, get_mfcc_vectors, get_codebook
from detect_word import get_obs, classify

models_file = "./models.p"

models = pickle.load(open(models_file, "rb"))
print(models)

options = ["go", "stop"]

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
    correct_pred = 0
    for _ in range(2000):
        choice = random.choice(options)
        label = choice
        index = random.randint(0, 649)
        while True:
            try:
                test_file = test[label][index]
                break
            except IndexError:
                index = random.randint(0, 649)
                continue
        vecs = get_mfcc_vectors([test_file])
        obs = get_obs(vecs, book)
        # print(obs)
        try:
            pred = classify(models, obs)
        except:
            print("Some exception occured due to div by zero, skipping")
            continue
        if pred == label:
            correct_pred += 1

    os.system('cls' if os.name == 'nt' else 'clear')    
    print("Accuracy: ", (correct_pred/2000)*100)
        

