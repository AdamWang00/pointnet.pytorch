import os
import json

room_name = "Bedroom3"
split_subdir_name = "test64"
testing_size = 64

split_filename_all = "all.json"
split_filename_train = "train.json"
split_filename_test = "test.json"

data_dir = "./data"
splits_dir = "./splits"
rooms_subdir = "Rooms"

filenames_all = os.listdir(os.path.join(data_dir, room_name, rooms_subdir))
filenames_test = filenames_all[0:testing_size]
filenames_train = filenames_all[testing_size:]

print(len(filenames_all), "files found")

splits_dir = os.path.join(splits_dir, room_name, split_subdir_name)
os.makedirs(splits_dir)
with open(os.path.join(splits_dir, split_filename_all), "w") as f:
    json.dump(filenames_all, f)
with open(os.path.join(splits_dir, split_filename_train), "w") as f:
    json.dump(filenames_train, f)
with open(os.path.join(splits_dir, split_filename_test), "w") as f:
    json.dump(filenames_test, f)