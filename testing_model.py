import pickle

with open("trainModel.pkl", "rb") as f:
    obj = pickle.load(f)

print(type(obj))