from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import csv
from os import walk

import math


def get_train_test_df(data_path: Path, num_of_users: int, test_ratio=0.3):

  def _normalize_user_id(data_df: pd.DataFrame):
    # User id normalization
    data_df["user"] = data_df["PARTICIPANT_ID"]
    users = np.unique(data_df["user"])
    for user_id, new_user_id in tqdm(zip(users, list(range(0, len(users))))):
      data_df.loc[data_df['user'] == user_id,'user'] = new_user_id

    data_df['user'] = data_df['user'].astype(int)

    user_column = data_df.pop('user')
    data_df.insert(0, 'user', user_column)

    return data_df
  
  train_dfs = []
  test_dfs = []
  i = 0
  for file_name in tqdm(next(walk(data_path))[2]):
    file_path = data_path / file_name
    try:
      df = pd.read_csv(file_path, delimiter='\t',quoting=csv.QUOTE_NONE, encoding = "ISO-8859-1", usecols = ['PARTICIPANT_ID','TEST_SECTION_ID','PRESS_TIME', 'RELEASE_TIME', 'LETTER', 'KEYCODE']) #delimiter='\t', quotechar="")
      split_idx = int(len(df) * (1. - test_ratio))
      train_df, test_df = df[:split_idx], df[split_idx:]
    except Exception as e:
      print(e)
      print(file_path)
      continue
    train_dfs.append(train_df)
    test_dfs.append(test_df)

    if i > num_of_users:
      break
    i += 1

  return _normalize_user_id(pd.concat(train_dfs, axis=0)), _normalize_user_id(pd.concat(test_dfs, axis=0))


def df_to_matrix(df: pd.DataFrame):
    matrix = [[[]  for _ in range(255)] for _ in range(len(np.unique(df["user"])))]
    for user_id, key_code, rt, pt  in zip(df["user"], df["KEYCODE"], df["RELEASE_TIME"], df["PRESS_TIME"]):
        if not math.isnan(key_code):
            key_code = int(key_code)
            matrix[user_id][key_code].append(rt - pt)

    return matrix


if __name__ == "__main__":
    train_df, test_df = get_train_test_df(data_path=Path("dataset/Keystrokes/files"), num_of_users=100)
    print(train_df.head())
    print(test_df.head())
