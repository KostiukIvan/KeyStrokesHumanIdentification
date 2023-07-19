from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import csv
from os import walk
import argparse


def get_all_data_in_df(data_path: Path, num_of_users: int):
  dfs = []
  i = 0
  for file_name in tqdm(next(walk(data_path))[2]):
    #print(f"Reading file : {file_name}")
    file_path = data_path / file_name
    try:
      df = pd.read_csv(file_path, delimiter='\t',quoting=csv.QUOTE_NONE, encoding = "ISO-8859-1") #, usecols = ['PARTICIPANT_ID','TEST_SECTION_ID','PRESS_TIME', 'RELEASE_TIME', 'KEYCODE']) #delimiter='\t', quotechar="")
    except Exception as e:
      print(e)
      print(file_path)
      continue
    dfs.append(df)

    if i > num_of_users:
      break
    i += 1

  return pd.concat(dfs, axis=0)

def to_horizontal_state(df: pd.DataFrame, num_of_strokes: int):
  # To horizontal state
  final_result = []
  for user_id in tqdm(np.unique(df["PARTICIPANT_ID"])):
    user_df = df[df["PARTICIPANT_ID"] == user_id]
    user_df = user_df.sort_values('PRESS_TIME')

    for idx in range(0, len(list(user_df["PRESS_TIME"])) - num_of_strokes + 1, num_of_strokes):
      press_time = list(user_df["PRESS_TIME"])[idx:idx + num_of_strokes]
      release_time = list(user_df["RELEASE_TIME"])[idx:idx + num_of_strokes]
      key_code = list(user_df["KEYCODE"])[idx:idx + num_of_strokes]

      final_result.append([user_id, *press_time, *release_time, *key_code])

  return pd.DataFrame(final_result, columns =['PARTICIPANT_ID', 
                                                *["press-"+str(i) for i in range(0, num_of_strokes)], 
                                                *["release-"+str(i) for i in range(0, num_of_strokes)], 
                                                *["key-"+str(i) for i in range(0, num_of_strokes)]
                                                ], dtype = float)

def normalize_user_id(data_df: pd.DataFrame):
  # User id normalization
  data_df["user"] = data_df["PARTICIPANT_ID"]
  users = np.unique(data_df["user"])
  num_classes = len(users)
  for user_id, new_user_id in tqdm(zip(users, list(range(0, num_classes)))):
    data_df.loc[data_df['user'] == user_id,'user'] = new_user_id

  data_df['user'] = data_df['user'].astype(int)

  user_column = data_df.pop('user')
  data_df.insert(0, 'user', user_column)

  return data_df

def generate_ppd_rpd_hd(data_df: pd.DataFrame, num_of_strokes: int):
  # Generation new data
  for i in range(1,num_of_strokes):
      data_df['PPD-'+str(i)] = data_df['press-'+str(i)] - data_df['press-'+str(i-1)]
      data_df['RPD-'+str(i)] = data_df['release-'+str(i)] - data_df['press-'+str(i-1)]

  for i in range(num_of_strokes):
      data_df['HD-'+str(i)] = data_df['release-'+str(i)] - data_df['press-'+str(i)]
  return data_df

def clean_data(data_df: pd.DataFrame, num_of_strokes: int):
  data_df.drop(columns=["press-" + str(i) for i in range(num_of_strokes)], inplace=True)
  data_df.drop(columns=["release-" + str(i) for i in range(num_of_strokes)], inplace=True)
  data_df = data_df.dropna()
  return data_df


def start(args: argparse.Namespace):
  df = get_all_data_in_df(data_path=args.data_path, num_of_users=args.num_of_users)
  data_df = to_horizontal_state(df=df, num_of_strokes=args.num_of_strokes)
  data_df = normalize_user_id(data_df=data_df)
  data_df = generate_ppd_rpd_hd(data_df=data_df, num_of_strokes=args.num_of_strokes)
  data_df = clean_data(data_df=data_df, num_of_strokes=args.num_of_strokes)

  #save data
  data_df.to_csv(args.save_path, sep='\t', index=False)
  read_df = pd.read_csv(args.save_path, delimiter='\t')
  print(read_df)

if __name__=="__main__":
  argParser = argparse.ArgumentParser()
  argParser.add_argument("-dp", "--data_path", type=Path, default="./dataset/Keystrokes/files", help="dataset path")
  argParser.add_argument("-sp", "--save_path", type=Path, default="./full_data.csv", help="dataset path")
  argParser.add_argument("-ns", "--num_of_strokes", type=int, default=200, help="num of strokes to use in split")
  argParser.add_argument("-nu", "--num_of_users", type=int, default=169000, help="num of strokes to use in split")
  args = argParser.parse_args()
  print("args=%s" % args)

  start(args)
