from pathlib import Path

import src.vd_model.loss_functions as loss_func
from src.vd_model.dataset import df_to_hd_matrix, df_to_rpd_matrix, get_train_test_df
from src.vd_model.model import get_id_of_the_most_probable_user


train_df, test_df = get_train_test_df(
    data_path=Path("dataset/Keystrokes/files"), num_of_users=100, test_ratio=0.5
)
loss_functions = {
    "kolomogorow_smirnow": loss_func.kolomogorow_smirnow,
}


def interfere_vd_model_only_with_HD_data():
    results = []
    train_hd_matrix = df_to_hd_matrix(train_df)
    test_hd_matrix = df_to_hd_matrix(test_df)

    for func_name, loss_f in loss_functions.items():
        success_rate = 0
        total_users = 0
        for user_id, sample in enumerate(test_hd_matrix):
            pred_id, _ = get_id_of_the_most_probable_user(
                train_hd_matrix, [sample], loss_f
            )
            total_users += 1
            success_rate += pred_id == user_id

        results.append(f"Func {func_name} success rate={success_rate / total_users}")

    print("====================== RESULTS on HD matrix ====================")
    for row in results:
        print(row)


def interfere_vd_model_only_with_RPD_data():
    results = []
    train_rpd_matrix = df_to_rpd_matrix(train_df)
    test_rpd_matrix = df_to_rpd_matrix(test_df)

    for func_name, loss_f in loss_functions.items():
        success_rate = 0
        total_users = 0
        for user_id, sample in enumerate(test_rpd_matrix):
            pred_id, _ = get_id_of_the_most_probable_user(
                train_rpd_matrix, [sample], loss_f
            )
            total_users += 1
            success_rate += pred_id == user_id

        results.append(f"Func {func_name} success rate={success_rate / total_users}")

    print("====================== RESULTS on RPD matrix ====================")
    for row in results:
        print(row)


def interfere_vd_model_only_HD_and_RPD_data():
    results = []

    train_hd_matrix = df_to_hd_matrix(train_df)
    test_hd_matrix = df_to_hd_matrix(test_df)

    train_rpd_matrix = df_to_rpd_matrix(train_df)
    test_rpd_matrix = df_to_rpd_matrix(test_df)

    for func_name, loss_f in loss_functions.items():
        success_rate = 0
        total_users = 0
        for user_id, (hd_sample, rpd_sample) in enumerate(
            zip(test_hd_matrix, test_rpd_matrix)
        ):
            hd_pred_id, hd_loss = get_id_of_the_most_probable_user(
                train_hd_matrix, [hd_sample], loss_f
            )
            rpd_pred_id, rpd_loss = get_id_of_the_most_probable_user(
                train_rpd_matrix, [rpd_sample], loss_f
            )
            if hd_loss < rpd_loss:
                pred_id = hd_pred_id
            else:
                pred_id = rpd_pred_id
            total_users += 1
            success_rate += pred_id == user_id

        results.append(f"Func {func_name} success rate={success_rate / total_users}")

    print("====================== RESULTS on HD & RPD matrix ====================")
    for row in results:
        print(row)


if __name__ == "__main__":
    interfere_vd_model_only_with_HD_data()
    interfere_vd_model_only_with_RPD_data()
    interfere_vd_model_only_HD_and_RPD_data()
