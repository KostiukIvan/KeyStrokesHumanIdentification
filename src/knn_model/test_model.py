from pathlib import Path
from src.knn_model.dataset import get_train_test_df, df_to_matrix
from src.knn_model.model import get_id_of_the_most_probable_user
import src.knn_model.loss_functions as loss_func
import numpy as np

def test_knn_model():
    train_df, test_df = get_train_test_df(data_path=Path("dataset/Keystrokes/files"), num_of_users=100, test_ratio=0.27)
    train_matrix = df_to_matrix(train_df)
    test_matrix = df_to_matrix(test_df)

    loss_functions = {"kolomogorow_smirnow": loss_func.kolomogorow_smirnow,
                      "t_test": loss_func.t_test,
                      "mann_whitney_u_test": loss_func.mann_whitney_u_test,
                      "kl_divergense_non_symetric": loss_func.kl_divergense_non_symetric,
                      "jennsen_shanon_symetric": loss_func.jennsen_shanon_symetric,}

    for func_name, loss_f in loss_functions.items():
        success_rate = 0
        total_users = 0
        for user_id, sample in enumerate(test_matrix):
            pred_id, _ = get_id_of_the_most_probable_user(train_matrix, [sample], loss_f)
            total_users += 1
            success_rate += (pred_id == user_id)

        print(f"Func {func_name} success rate={success_rate / total_users}")

    assert False

    
# results on 100 users
# Func kolomogorow_smirnow success rate=0.7450980392156863
# Func t_test success rate=0.058823529411764705
# Func mann_whitney_u_test success rate=0.00980392156862745
# Func kl_divergense_non_symetric success rate=0.0784313725490196
# Func jennsen_shanon_symetric success rate=0.0
