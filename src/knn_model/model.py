from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import csv
from os import walk
import argparse
from src.knn_model.utils import get_press_release_data
import math
import src.knn_model.loss_functions as loss_func


def add_new_empty_user(matrix): 
    print(f"Your user id will be {len(matrix)}")
    matrix.append([[]  for _ in range(255)])
    return matrix, len(matrix) - 1 

def add_train_data_to_user_from_stream(matrix, user_id): 
    print("********************************************************************************")
    print("********************************************************************************")
    print("***** Now you are asked to type some text (without pressing enter) just text. **")
    print("*****               When you finish please press: *** Esc ***                 **")
    print("********************************************************************************")
    print("********************************************************************************")   
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


def read_test_text_and_print_similarity_prob(matrix, user_id):
    print("********************************************************************************")
    print("********************************************************************************")
    print("***** Now type different text. We will use this as a sample to find your USER_ID")
    print("*****               When you finish please press: *** Esc ***                 **")
    print("********************************************************************************")
    print("********************************************************************************")

    press_events, release_events = get_press_release_data()
    sample = build_sample_vector_based_on_press_release_data(press_events, release_events)
    probability = interference(matrix[user_id], sample, loss_func.kolomogorow_smirnow)

    print("********************************************************************************")
    print("********************************************************************************")
    print(f"Probability that these are the same people = {probability}")
    print("********************************************************************************")
    print("********************************************************************************")


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


def get_id_of_the_most_probable_user(matrix, sample, loss_function = loss_func.t_test):
    losses = [10000] * len(matrix)
    for user_id, row in enumerate(matrix):
        losses[user_id] = interference(sample[0], row, loss_function)

    # min loss
    min_id = 0
    min_loss = losses[0]
    for user_id, loss in enumerate(losses):
       if loss < min_loss:
          min_loss = loss
          min_id = user_id
    return min_id, min_loss

def interference(x_sample, y_sample, loss_function = loss_func.kolomogorow_smirnow):
    loss = 0.
    n = 0
    for x_dist, y_dist in zip(x_sample, y_sample):
        if len(x_dist) > 0  and len(y_dist) > 0:
            l, _= loss_function(x_dist, y_dist)
            loss += l
            n += 1
    if loss > 0:
        loss /= n
        return loss
    else:
        return 100000