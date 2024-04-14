import argparse
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score
import os

from scripts import scripts


def main():
       
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=18,
                        help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=30,
                        help='Count of epoch for train model')
    parser.add_argument('-d', '--data', type=str, default='data',
                        help='The name of the folder, that will be to created to unpacking data there')
    parser.add_argument('-s', '--save', type=str, default='weights',
                        help='The name of the folder, that will be to created to save models')
    parser.add_argument('-n', '--name', type=str, default='efficientnet_v2_m__size_380__crop__aug__adamw__lr_0001_10_01',
                        help='Template of model names')
    args = parser.parse_args()

    # create save folder
    if not os.path.isdir(args.save):
        os.mkdir(f'{args.save}')

    annotation_train = f'{args.data}/kaggle_retina/labels_train.csv'
    annotation_val = f'{args.data}/kaggle_retina/labels_val.csv'
    img_train = f'{args.data}/kaggle_retina/train'
    img_val = f'{args.data}/kaggle_retina/val'

    # validate on train data only for temp
    dataset = {'train': scripts.CustomImageDataset(annotation_train, img_train, True, scripts.augmentation()),
                'val': scripts.CustomImageDataset(annotation_val, img_val, True, scripts.transforms())}
    dataset_sizes = {x: len(dataset[x]) for x in ['train', 'val']}
    dataloader = {x: torch.utils.data.DataLoader(dataset[x], batch_size=args.batch_size,
                                                shuffle=True)
            for x in ['train', 'val']}


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_v2_m(weights='IMAGENET1K_V1')
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), amsgrad=True, lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_model_params_path = f'{args.save}/{args.name}__best.pt'
    last_model_params_path = f'{args.save}/{args.name}__last.pt'

    best_f1 = 0.0

    pred_list = {'train': [], 'val': []}
    label_list = {'train': [], 'val': []}

    for epoch in range(args.epochs):
        print(f'Epoch {epoch}/{args.epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode

            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for inputs, labels, _ in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = torch.flatten(model(inputs))
                    loss = criterion(outputs, labels.float())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

                pred_list[phase] += [round(x) for x in outputs.tolist()]
                label_list[phase] += labels.tolist()

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_f1 = f1_score(label_list[phase], pred_list[phase], average='macro')

            print(f'{phase} Loss: {epoch_loss:.4f} F1: {epoch_f1:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                torch.save(model.state_dict(), best_model_params_path)

            torch.save(model.state_dict(), last_model_params_path)

            # zeroed labels
            pred_list = {'train': [], 'val': []}
            label_list = {'train': [], 'val': []}

        print()

    print(f'Best F1: {best_f1:4f}')

    return model

if __name__ == '__main__':
    main()