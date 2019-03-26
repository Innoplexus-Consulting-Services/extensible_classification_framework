import csv
import json
import random
import time
import os
import argparse


fieldnames = ("label","review")
json_array = []

def to_json(input_file, output_destination, sep = '\t', split_ratio = 0.8):
    """
    input_file: file where label and reviews are stored as delimited file.
    output_destination: folder where the train and test split will be written.
    to convert the input csv file to json and split in to train and test file.
    """
    if os.path.isdir(output_destination):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        accumlator = []
        with open(input_file) as csvfile:
            reader = csv.reader(csvfile, delimiter = str(sep))
            for row in reader:
                accumlator.append(str(json.dumps({'label' : row[0], 'review' : row[1]})))
        # print(len(accumlator), int(len(accumlator)*split_ratio))

        # randomly shuffling dataset.
        random.shuffle(accumlator)
        train_split = accumlator[:int(len(accumlator)*split_ratio)]
        test_split = accumlator[int(len(accumlator)*split_ratio):]

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
    """
    Usage : python preprocess.py --input_file /data/extensibleFramework/extensibleFramework/data/crude/small_data.tsv --output_destination /data/extensibleFramework/extensibleFramework/data/processed/ --sep "tab"
    """
    seperator = {"tab": "\t", "comma":",", "colon":":", "semicolon":";"}
    parser = argparse.ArgumentParser(description='Preprocess data for text classification')
    parser.add_argument('--input_file', help='Input file having label and review seperated by delimiter.', required = True)
    parser.add_argument('--output_destination',
                        help='Destination folder where preprocessed file will be written.', required = True)
    parser.add_argument('--sep', help='delimiter according to file structure. options: "tab" or "comma" or "colon" or "semicolon"', required = True)
    parser.add_argument('--split_ratio', help='split ratio of the train file, any float value', default = 0.7)
    args = parser.parse_args()
    
    to_json(args.input_file, args.output_destination, seperator[args.sep], args.split_ratio)