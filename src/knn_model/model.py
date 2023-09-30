from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import csv
from os import walk
import argparse
from src.knn_model.utils import get_press_release_data
import math

def _get_mean_values_of_matrix(matrix):
    for user_id in range(len(matrix)):
       for key_id in range(len(matrix[user_id])):
          matrix[user_id][key_id] = np.mean(matrix[user_id][key_id])
    return matrix

def get_matrix(df: pd.DataFrame):
    matrix = [[[]  for _ in range(255)] for _ in range(len(np.unique(df["user"])))]
    for user_id, key_code, rt, pt  in zip(df["user"], df["KEYCODE"], df["RELEASE_TIME"], df["PRESS_TIME"]):
        if not math.isnan(key_code):
            key_code = int(key_code)
            matrix[user_id][key_code].append(rt - pt)

    return matrix

def add_new_empty_user(matrix): 
    print(f"Your user id will be {len(matrix)}")
    matrix.append([[]  for _ in range(255)])
    return matrix

def add_data_to_user(matrix, user_id):    
    assert 0 <= user_id < len(matrix)

    press_events, release_events = get_press_release_data()

    for i, press_event in enumerate(press_events):
        press_key, press_timestamp = press_event
        hd = None
        for j, release_event in enumerate(release_events[i:]):
            release_key, release_timestamp = release_event
            if press_key == release_key and j < 5:
               hd = release_timestamp - press_timestamp
               break
        if hd:
            matrix[user_id][press_key].append(hd)
        else:
            print(f"ERROR while computing HD")
    return matrix


def predict_real_user_id(matrix):
    press_events, release_events = get_press_release_data()
    sample = build_sample_vector_based_on_press_release_data(press_events, release_events)
    transf_matrix = _get_mean_values_of_matrix(matrix)
    transf_sample = _get_mean_values_of_matrix([sample])
    min_id, min_loss = get_id_of_user_based_on_sample(transf_matrix, transf_sample)
    print("====" * 10)
    print("====" * 10)
    print("====" * 10)
    print(f"Most probably you are the user id={min_id}, loss={min_loss}")
    print("====" * 10)
    print("====" * 10)
    print("====" * 10)


def build_sample_vector_based_on_press_release_data(press_events, release_events):
    sample = [[]  for _ in range(255)]
    for i, press_event in enumerate(press_events):
        press_key, press_timestamp = press_event
        hd = None
        for j, release_event in enumerate(release_events[i:]):
            release_key, release_timestamp = release_event
            if press_key == release_key and j < 5:
               hd = release_timestamp - press_timestamp
               break
        if hd:
          sample[press_key].append(hd)
        else:
            print(f"ERROR while computing HD")
    return sample

def get_id_of_user_based_on_sample(matrix, sample):
    losses = [10000] * len(matrix)
    for user_id, row in enumerate(matrix):
        loss = 0.
        n = 0
        for x, y in zip(sample[0], row):
          if x > 0 and y > 0:
             loss += (x - y) ** 2
             n += 1
        if loss > 0:
          loss /= n
          losses[user_id] = loss
    # min loss
    min_id = 0
    min_loss = losses[0]
    for user_id, loss in enumerate(losses):
       if loss < min_loss:
          min_loss = loss
          min_id = user_id
    return min_id, min_loss

