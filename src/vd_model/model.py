import src.vd_model.loss_functions as loss_func



def build_sample_vector_based_on_press_release_data(press_events, release_events):
    sample = [[] for _ in range(255)]
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


def get_id_of_the_most_probable_user(matrix, sample, loss_function=loss_func.kolomogorow_smirnow):
    losses = [100_000] * len(matrix)
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


def interference(x_sample, y_sample, loss_function=loss_func.kolomogorow_smirnow):
    loss = 0.0
    n = 0
    for x_dist, y_dist in zip(x_sample, y_sample):
        if len(x_dist) > 3 and len(y_dist) > 3:
            l = loss_function(x_dist, y_dist)
            loss += l
            n += 1

    if loss > 0:
        loss /= n
        return loss
    else:
        return 100_000
