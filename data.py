import zipfile
import shutil
import pandas as pd
import numpy as np
import os
import argparse

def main():
  dir = os.path.abspath(os.getcwd())

  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--zip_main', type=str, 
                      default=f'{dir}/diabetic-retinopathy-classification-f1-score-4.zip',
                      help='Path to zip file of competition')
  parser.add_argument('-a', '--zip_2015', type=str, default=f'{dir}/train_image_resized.zip',
                      help='Path to zip file of additional data')
  parser.add_argument('-d', '--directory', type=str, default='data',
                      help='The name of the folder, that will be to created to unpacking data there')
  args = parser.parse_args()

  # unpacking data
  print('Step 1: start data preparation')
  directory_to_extract_to = args.directory
  with zipfile.ZipFile(args.zip_main, 'r') as zip_ref:
      zip_ref.extractall(directory_to_extract_to)

  directory_to_extract_to = args.directory
  with zipfile.ZipFile(args.zip_2015, 'r') as zip_ref:
      zip_ref.extractall(directory_to_extract_to)
  print('Step 1: data preparation successful')
  print('-'*25)

  # labels
  print('Step 2: start labels preparation')
  train_table_main = pd.read_csv(f'{args.directory}/kaggle_retina/train.csv')
  train_table_2015 = pd.read_csv(f'{args.directory}/train_image_resized/trainLabels.csv')
  train_table_main['path'] = train_table_main['id_code'] + '.png'
  train_table_2015['path'] = train_table_2015['image'] + '.jpeg'
  train_table_2015['is_val'] = 0
  train_table_2015.rename(columns={"image": "id_code", "level": "diagnosis", "path": "path", "is_val": "is_val"},
                        inplace=True)
  print('Step 2: labels preparation successful')
  print('-'*25)

  # train-test split
  print('Step 3: start train-test split')
  print('Substep: split')
  np.random.seed(10)
  train_table_main['is_val'] = np.random.choice([1, 0], size=len(train_table_main), p=[.2, .8])

  print('Substep: prepare validation set')
  os.mkdir(f'{args.directory}/kaggle_retina/val')
  for i in train_table_main[train_table_main['is_val'] == 1]['path']:
      src_dir=f"{args.directory}/kaggle_retina/train/{i}"
      dst_dir=f"{args.directory}/kaggle_retina/val/{i}"
      shutil.copy(src_dir,dst_dir)
      os.remove(f"{args.directory}/kaggle_retina/train/{i}")

  labels_val = train_table_main[train_table_main['is_val'] == 1][['path', 'diagnosis']]
  labels_val.to_csv(path_or_buf=f'{args.directory}/kaggle_retina/labels_val.csv', header=False, index=False)

  print('Substep: prepare train set with data addition')
  # add data to train for balancing
  samples_2015 = {0: list(train_table_2015[train_table_2015['diagnosis'] == 0]['path'][:151]),
                  2: list(train_table_2015[train_table_2015['diagnosis'] == 2]['path'][:515]),
                  1: list(train_table_2015[train_table_2015['diagnosis'] == 1]['path'][:825]),
                  4: list(train_table_2015[train_table_2015['diagnosis'] == 4]['path'][:708]),
                  3: list(train_table_2015[train_table_2015['diagnosis'] == 3]['path'][:873])}

  # move to train set
  for idx in range(5):
    for i in samples_2015[idx]:
      src_dir=f"{args.directory}/train_image_resized/train/{i}"
      dst_dir=f"{args.directory}/kaggle_retina/train/{i}"
      shutil.copy(src_dir,dst_dir)
    # union
    train_table_main = pd.concat([train_table_main,
                                  train_table_2015[train_table_2015['path'].isin(samples_2015[idx])]],
                                 ignore_index=True)
  # train
  labels_train = train_table_main[train_table_main['is_val'] == 0][['path', 'diagnosis']]
  labels_train.to_csv(path_or_buf=f'{args.directory}/kaggle_retina/labels_train.csv', header=False, index=False)
  print('Step 3: train-test split successful')
  print('Finish')

if __name__ == '__main__':
    main()