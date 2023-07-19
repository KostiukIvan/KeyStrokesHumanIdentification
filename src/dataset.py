import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split

def get_train_and_test_df(data_path: Path, num_of_users: int, test_size):
    data_df = pd.read_csv(data_path, delimiter='\t')
    train_data = []
    test_data = []
    for user in tqdm(np.unique(data_df["user"])[:num_of_users]):
        user_df = data_df[data_df["user"] == user]
        if len(user_df) > 1:
            train, test = train_test_split(user_df, test_size=test_size)
            train_data.append(train)
            test_data.append(test)

    train_df = pd.concat(train_data, ignore_index=True)
    valid_df = pd.concat(test_data, ignore_index=True)

    print(f"\nTrain dataset len {len(train_df)} and valid len {len(valid_df)}")
    return train_df, valid_df


if __name__=="__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-dp", "--data_path", type=Path, default="./dataset/full_data.csv", help="dataset path")
    argParser.add_argument("-ts", "--test_size", type=int, default=0.2, help="test size percentage in 0.")
    argParser.add_argument("-nu", "--num_of_users", type=int, default=169000, help="num of strokes to use in split")
    args = argParser.parse_args()
    print("args=%s" % args)

    train_df, valid_df = get_train_and_test_df(data_path=args.data_path, num_of_users=args.num_of_users, test_size=args.test_size)
