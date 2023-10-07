import time
from src.vd_model.model import build_sample_vector_based_on_press_release_data, interference, get_id_of_the_most_probable_user
import src.vd_model.loss_functions as loss_func
from pynput import keyboard


def get_press_release_data():
    press_events = []

    def on_press(key):
        if type(key) is keyboard.KeyCode:
            value = ord(key.char)
            press_events.append((value, time.time() * 1000))
        else:
            pass
            # TODO: all non-alphanumeric keyboard key are skipped. Temporary

    release_events = []

    def on_release(key):
        if type(key) is keyboard.KeyCode:
            value = ord(key.char)
            release_events.append((value, time.time() * 1000))
        else:
            pass
            # TODO: all non-alphanumeric keyboard key are skipped. Temporary
        if key == keyboard.Key.esc:
            return False

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    return press_events, release_events



def add_new_empty_user(matrix):
    print(f"Your user id will be {len(matrix)}")
    matrix.append([[] for _ in range(255)])
    return matrix, len(matrix) - 1


def add_train_data_to_user_from_stream(matrix, user_id):
    print("**************************************************************************")
    print("**************************************************************************")
    print("* Now you are asked to type some text (without pressing enter) just text *")
    print("*              When you finish please press: *** Esc ***                **")
    print("**************************************************************************")
    print("**************************************************************************")
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


def read_test_text_and_print_similarity_prob(matrix):
    print("**************************************************************************")
    print("**************************************************************************")
    print("Now type different text. We will use this as a sample to find your USER_ID")
    print("*               When you finish please press: *** Esc ***                *")
    print("**************************************************************************")
    print("**************************************************************************")

    press_events, release_events = get_press_release_data()
    sample = build_sample_vector_based_on_press_release_data(
        press_events, release_events
    )
    #probability = interference(matrix[user_id], sample, loss_func.kolomogorow_smirnow)
    user_id, prob = get_id_of_the_most_probable_user(matrix, [sample], loss_func.kolomogorow_smirnow)

    print("**************************************************************************")
    print("**************************************************************************")
    print(f"It's user={user_id} with probability={prob}")
    print("**************************************************************************")
    print("**************************************************************************")
