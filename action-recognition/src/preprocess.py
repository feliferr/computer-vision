from operator import index
import pandas as pd
import joblib
import os
import argparse
import numpy as np

from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer

def main(input_path, output_path):
    all_paths = os.listdir(input_path)
    folder_paths = [folder_path for folder_path in all_paths if os.path.isdir(os.path.join(input_path, folder_path))]
    print(f"Folder paths: {folder_paths}")
    print(f"Number of folders :{len(folder_paths)}")

    create_labels = ["basketball", "boxing", "chess"]

    data = pd.DataFrame()

    image_formats = ['jpg', 'JPG', 'PNG', 'png']
    labels = []
    counter = 0
    for i, folder_path in tqdm(enumerate(folder_paths), total=len(folder_paths)):
        if folder_path not in create_labels:
            continue
        image_paths = os.listdir(os.path.join(input_path, folder_path))
        label = folder_path

        for image_path in image_paths:
            if image_path.split('.')[-1] in image_formats:
                data.loc[counter, 'image_path'] = f"{input_path}/{folder_path}/{image_path}"
                labels.append(label)
                counter += 1

    labels = np.array(labels)
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    print(labels.shape)
    if len(labels[0]) == 1:
        for i in range(len(labels)):
            index = labels[i]
            data.loc[i, 'target'] = int(index)
    elif len(labels[0]) > 1:
        for i in range(len(labels)):
            index = np.argmax(labels[i])
            data.loc[i, 'target'] = int(index)
    
    # shuffle the dataset
    data = data.sample(frac=1).reset_index(drop=True)

    print(f"Number of classes: {len(lb.classes_)}")
    print(f"Total instances: {len(data)}")

    # save to csv
    data.to_csv(os.path.join(input_path,"data.csv"), index=False)

    # saving the binarized labels
    print("Saving the binarized labels in pickled file")
    joblib.dump(lb, os.path.join(output_path, "lb.pkl"))

    print(data.head(5))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_path", required=True, help="Path to the root folder containing dataset images")
    ap.add_argument("-o", "--output_path", required=True, help="Path to save the preprocessed data")

    args = ap.parse_args()
    arguments = args.__dict__
    
    main(arguments)