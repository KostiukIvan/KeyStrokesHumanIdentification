from pathlib import Path

from src.vd_model.dataset import df_to_hd_matrix, get_train_test_df, df_to_rpd_matrix
from src.vd_model.utils import (
    add_new_empty_user,
    add_train_data_to_user_from_stream,
    read_test_text_and_print_similarity_prob,
)

if __name__ == "__main__":
    train_df, _ = get_train_test_df(
        data_path=Path("dataset/Keystrokes/files"),
        num_of_users=100,
    )
    matrix_hd = df_to_hd_matrix(train_df)

    matrix_hd, user_id = add_new_empty_user(matrix=matrix_hd)
    add_train_data_to_user_from_stream(matrix=matrix_hd, user_id=user_id)
    read_test_text_and_print_similarity_prob(matrix=matrix_hd)
