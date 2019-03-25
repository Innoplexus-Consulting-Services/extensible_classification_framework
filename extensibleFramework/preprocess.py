import csv
import json
import random
import time
import os
import argparse


fieldnames = ("label","review")
json_array = []

def to_json(input_file, output_destination, sep = "\t", split_ratio = 0.8):
    """
    to convert the input csv file to json and split in to train and test file.
    """
    if os.path.isdir(output_destination):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        csvfile = open(input_file, 'r')
        reader = csv.DictReader(csvfile, fieldnames, delimiter=sep)
        for row in reader:
            json_array.append(row)

        # randomly shuffling dataset.
        random.shuffle(json_array)
        train_split = json_array[:int(len(json_array)*split_ratio)]
        test_split = json_array[int(len(json_array)*split_ratio):]

        # writtng train file
        train_file =open(os.path.join(output_destination,timestr+"_train.json"), "w")
        for each_entry in train_split:
            train_file.write(str(each_entry)+"\n")
        train_file.close()

        # writtng test file
        test_file =open(os.path.join(output_destination,timestr+"_test.json"), "w")
        for each_entry in test_split:
            test_file.write(str(each_entry)+"\n")
        test_file.close()

        print("Train file is written at : ", os.path.join(output_destination,timestr+"_train.json"))
        print("Test file is written at : ", os.path.join(output_destination,timestr+"_test.json"))

    else:
        print(output_destination+" No such folder found, terminating") 
        # this is to avoid run if the user has provided a file path

    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Preprocess data for text classification')
    parser.add_argument('--input_file', help='Input file having label and review seperated by delimiter.', required = True)
    parser.add_argument('--output_destination',
                        help='Destination folder where preprocessed file will be written.', required = True)
    parser.add_argument('--sep', help='delimiter according to file structure.', required = True)
    parser.add_argument('--split_ratio', help='split ratio of the train file.', default = 0.7)
    args = parser.parse_args()
    to_json(args.input_file, args.output_destination, args.sep, args.split_ratio)