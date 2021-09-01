from src.test import predict
import apache_beam as beam
import click
import io
import logging
import pathlib
import joblib
import numpy as np
import os
import cv2
import logging
import src.cnn_models
import torch
import albumentations
import json

from PIL import Image
from datetime import datetime
from apache_beam.io import ReadFromText
from apache_beam.io.gcp.gcsio import GcsIO
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions


os.environ['GOOGLE_CLOUD_DISABLE_GRPC'] = 'true'

def singleton(cls):
    instances = {}

    def getinstance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return getinstance

def download_gs_file(gs_file_path):
    buf = GcsIO().open(gs_file_path, mode="rb")
    bio = io.BytesIO(buf.read())

    filename = gs_file_path.split('/')[-1]

    with open(filename, "wb") as output_file:
        output_file.write(bio.getbuffer())
    
    print(f"---- file {gs_file_path} downloaded to {filename}")
    return filename

@singleton
class Model:

    def __init__(self, path_model, path_label_bin, device):
        self.aug = albumentations.Compose([
            albumentations.Resize(224, 224),
        ])
        self.label_binarizer = joblib.load(download_gs_file(path_label_bin))
        self.device = device

        self.model = src.cnn_models.SimpleCNN(len(self.label_binarizer.classes_)).to(device)
        self.model.load_state_dict(torch.load(download_gs_file(path_model)))
    
    def predict(self, input_path_video, output_path):
        cap = cv2.VideoCapture(input_path_video)

        if cap.isOpened() == False:
            print(f"Error while trying to read video '{input_path_video}'")
            return ''
        
        # get the frame width and height
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # define codec and create VideoWriter
        out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

        actions = []

        while cap.isOpened():
            # taking each frame from video
            ret, frame = cap.read()
    
            if ret == True:
                self.model.eval()
                with torch.no_grad():
                    # convert to PIL RGB format before predictions
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    pil_image = self.aug(image=np.array(pil_image))['image']
                    pil_image = np.transpose(pil_image, (2, 0, 1)).astype(np.float32)
                    pil_image = torch.tensor(pil_image, dtype=torch.float).to(self.device)
                    pil_image = pil_image.unsqueeze(0)

                    outputs_model = self.model(pil_image)
                    _, preds = torch.max(outputs_model.data, 1)
                
                # writing the predicted label to video

                # ============= ONLY TO BE USED FOR DEMOSTRATION
                # cv2.putText(frame, 
                #             self.label_binarizer.classes_[preds], 
                #             (10, 30), 
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             0.9, 
                #             (0, 200, 0), 
                #             2
                #             )
                # out.write(frame)
                # ++++++++++++++++++++++++++++++++++++++++++++++++
                actions.append(self.label_binarizer.classes_[preds])

                # cv2.imshow('image', frame)
                
                # press 'q' to exit
                # if cv2.waitKey(27) & 0xFF == ord('q'):
                #     break
            else:
                break
        
        # release VideoCapture()
        cap.release()

        # close all frames and video windows
        # cv2.destroyAllWindows()

        json_content = {"actions":list(set(actions))}
        json_object = json.dumps(json_content)
  
        # Writing to output_path
        with open(output_path, "w") as outfile:
            outfile.write(json_object)

        print(f"Prediction saved at '{output_path}' ")

        return output_path

def load_and_predict(path, path_model, label_bin, output_path, device):
    model = Model(path_model, label_bin, device)
    
    # TODO improve this
    l = path.split("/")
    media_id = l[len(l)-2]

    video_file_path = download_gs_file(path)
    video_output = f"processed_{media_id}.json"

    return model.predict(video_file_path, video_output)

def store_in_gs(path, output):
    os.system(f"gsutil -m cp {path} {output}")
    os.system(f"rm -rf {path}")


@click.command()
@click.option("--job-name")
@click.option("--path-model", "path_model")
@click.option("--label-bin", "label_bin")
@click.option("--input", "input_path")
@click.option("--output", "output_path")
@click.option("--device", "device")
@click.option("--max-num-workers", default=None, type=int)
@click.option("--worker-machine-type", "worker_machine_type")
@click.option("--local", is_flag=False)
@click.option("--project-id", envvar="GCP_PROJECT_ID")
@click.option("--region", envvar="GCP_REGION", default="us-west1")
def run(
    job_name,
    path_model,
    label_bin,
    input_path,
    output_path,
    device,
    max_num_workers,
    worker_machine_type,
    local,
    project_id,
    region  
):
    if job_name is None:
        job_name = f"video-action-recognition-classifier-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    click.echo(f"Running job: {job_name}")

    output_path = os.path.join(output_path, job_name)
    if local:
        # Execute pipeline in your local machine.
        runner_options = {
            "runner": "DirectRunner",
        }
    else:
        runner_options = {
            "runner": "DataflowRunner",
            "temp_location": os.path.join(output_path, "temp_location"),
            "staging_location.": os.path.join(output_path, "staging_location"),
            "machine_type": worker_machine_type,
            "max_num_workers": max_num_workers,
        }

    options = PipelineOptions(
        project=project_id,
        job_name=job_name,
        region=region,
        **runner_options
    )
    options.view_as(SetupOptions).save_main_session = True
    options.view_as(SetupOptions).setup_file = os.path.join(
        pathlib.Path(__file__).parent.absolute(), "..", "setup.py")
    
    with beam.Pipeline(options=options) as p:
        (
            p
            |"source" >> ReadFromText(input_path)
            |"load_and_predict" >> beam.Map(load_and_predict, path_model, label_bin, output_path, device)
            |"store_in_gs" >> beam.Map(store_in_gs, output_path)
        )

if __name__== "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()