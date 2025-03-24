import pickle

def load_dataset(input_file):
    with open(input_file, 'rb') as f:
        return pickle.load(f)