import os
from os import listdir
from os.path import isfile, join
import argparse
import pandas as pd

import torch
from torchvision import models
import torch.nn as nn

from scripts import scripts


def main():

    dir = os.path.abspath(os.getcwd())
    res = f'{os.path.abspath(os.getcwd())}/results/images'

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, 
                      default=f'{dir}/weights/efficientnet_v2_m__size_380__crop__aug__adamw__lr_0001_10_01__best.pt',
                      help='Path to model weights')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    files = {'path': [f for f in listdir(res) if isfile(join(res, f))]}
    annotation = pd.DataFrame(data=files)
    annotation['diagnosis'] = 0
    annotation.to_csv(path_or_buf=f'{dir}/results/result.csv', index=False)
    annotation = f'{dir}/results/result.csv'

    dataset = scripts.CustomImageDataset(annotation, res, True, scripts.transforms())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False)
    
    print('Model Initialization')
    model = models.efficientnet_v2_m(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    model.eval()

    # initialize dictionary
    keyList = ["name", "pred"]
    pred_dict = {}
    for i in keyList:
        pred_dict[i] = []
    
    print('Start Predict')
    for input, label, img_name in dataloader:
        input = input.to(device)

        with torch.set_grad_enabled(False):
            # get predict
            outputs = torch.flatten(model(input))

        pred_dict['name'].append(img_name[0])
        pred_dict['pred'].append([round(x) for x in outputs.tolist()][0])
    
    # save results
    result = pd.DataFrame(pred_dict)
    result.to_csv(path_or_buf=f'{dir}/results/result.csv', header=False, index=False)
    print(f'Finish! Results has been saved at "results/result.csv"')


if __name__ == '__main__':
    main()