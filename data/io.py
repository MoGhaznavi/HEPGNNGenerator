import os, pickle

def load_pickle_file(file_name, load_path):
    full_path = os.path.join(load_path, file_name)
    with open(full_path, 'rb') as pickle_file:
        file = pickle.load(pickle_file)
    print(f"Data successfully loaded from {full_path}")
    return file

def save_as_pickle(file_name, save_path, data_dict):
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, file_name)
    with open(full_path, 'wb') as file:
        pickle.dump(data_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Data saved to {full_path}")