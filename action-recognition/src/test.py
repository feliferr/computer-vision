import torch
import numpy as np
import argparse
import joblib
import cv2
import torch.nn as nn
import torch.nn.functional as F
import time
import cnn_models
import albumentations

from torchvision.transforms import transforms
from torch.utils.data import Dataset, dataloader
from PIL import Image


def predict(model, aug, label_binarizer, input, output, device):
    cap = cv2.VideoCapture(input)

    if cap.isOpened() == False:
        print(f"Error while trying to read video '{input}'")
    
    # get the frame width and height
    frame_width = int(cap.get(cv2.cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.cv2.CAP_PROP_FRAME_HEIGHT))

    # define codec and create VideoWriter
    out = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    while cap.isOpened():
        # taking each frame from video
        ret, frame = cap.read()
 
        if ret == True:
            model.eval()
            with torch.no_grad():
                # convert to PIL RGB format before predictions
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pil_image = aug(image=np.array(pil_image))['image']
                pil_image = np.transpose(pil_image, (2, 0, 1)).astype(np.float32)
                pil_image = torch.tensor(pil_image, dtype=torch.float).to(device)
                pil_image = pil_image.unsqueeze(0)

                outputs_model = model(pil_image)
                _, preds = torch.max(outputs_model.data, 1)
            
            # writing the predicted label to video
            cv2.putText(frame, 
                        label_binarizer.classes_[preds], 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, 
                        (0, 200, 0), 
                        2
                        )
            # cv2.imshow('image', frame)
            out.write(frame)

            # press 'q' to exit
            # if cv2.waitKey(27) & 0xFF == ord('q'):
            #     break
        else:
            break
    
    # release VideoCapture()
    cap.release()

    # close all frames and video windows
    cv2.destroyAllWindows()
    print(f"Prediction saved at '{output}' ")


def main(path_model, label_bin, input, output, device):

    # load the model and the binarizer
    lb = joblib.load(label_bin)
    print('Loaded label binarizer!')

    model = cnn_models.SimpleCNN().to(device)
    model.load_state_dict(torch.load(path_model))

    print('Loaded model and its state_dict!')

    aug = albumentations.Compose([
        albumentations.Resize(224, 224),
    ])

    # creating video name output to be saved
    video_predicted_name = f"{input.split('/')[-1].split('.')[0]}_predicted.mp4"
    video_output = f"{output}/{video_predicted_name}"

    # making action predictions
    predict(model, aug, lb, input, video_output, device)



if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--path-model', required=True,help="path to trained serialized mode")
    ap.add_argument('-l', '--label-bin', required=True,help="path to label binarizer")
    ap.add_argument('-i', '--input', required=True,help="path to the input video")
    ap.add_argument('-o','--output', required=True, type=str,help="folder path to save the prediction output")
    ap.add_argument('-d', '--device', default="cpu:0", help="Device to be used in the prediction")
    args = ap.parse_args()

    arguments = args.__dict__

    main(**arguments)

