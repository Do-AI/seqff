import argparse
from glob import glob
import numpy as np
import pandas as pd


def get_file_id(file):
    return file.split('/')[-1].split('.')[0]


def parse_normalized_bincounts(file_paths):
    X = {}
    for file in file_paths:
        file_id = get_file_id(file)
        with open(file, 'r') as f:
            lines = f.read().split('\n')[:-1]
        data = [float(i) for i in lines]
        X[file_id] = data
    return X


def parse_groundtruth(file_path):
    df = pd.read_csv(file_path, header=None).set_index(0)
    y = {}
    for idx, row in df.iterrows():
        file_id = idx.split('.')[0]
        y[file_id] = float(row[1])
    return y


def make_dataset(X):
    file_ids = []
    _X = []
    for file_id, value in X.items():
        file_ids.append(file_id)
        _X.append(value)
    _X = np.array(_X)
    return file_ids, _X


def make_datasets(X, y):
    file_ids = []
    _X = []
    _y = []
    for file_id, value in y.items():
        file_ids.append(file_id)
        _y.append(value)
        _X.append(X[file_id])
    _X = np.array(_X)
    _y = np.array(_y)

    return file_ids, _X, _y


def save_numpy(filepath, npy):
    np.save(filepath, npy)


def save_file_ids(filepath, file_ids):
    with open(filepath, 'w') as f:
        f.write('\n'.join(file_ids))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='normalized bincounts directory', default='')
    parser.add_argument('--gt_path', type=str, help='groundtruth path', default=None)
    parser.add_argument('--output_path', type=str, help='save directory', default='')
    args = parser.parse_args()

    file_paths = glob(args.input_path + '*')
    if args.gt_path is None:
        X = parse_normalized_bincounts(file_paths)
        file_ids, X = make_dataset(X)
    else:
        X = parse_normalized_bincounts(file_paths)
        y = parse_groundtruth(args.gt_path)
        file_ids, X, y = make_datasets(X, y)
        save_numpy(args.output_path + 'y', y)

    save_numpy(args.output_path + 'X', X)
    save_file_ids(args.output_path + 'file_ids.txt', file_ids)


if __name__ == '__main__':
    main()
