from pathlib import Path
from src.knn_model.dataset import get_train_test_df, df_to_matrix
from src.knn_model.model import add_new_empty_user, add_train_data_to_user_from_stream, read_test_text_and_print_similarity_prob


if __name__ == "__main__":
  train_df, _ = get_train_test_df(data_path=Path("dataset/Keystrokes/files"), num_of_users=100, )
  matrix = df_to_matrix(train_df)

  matrix, user_id = add_new_empty_user(matrix=matrix)
  add_train_data_to_user_from_stream(matrix=matrix, user_id=user_id)
  read_test_text_and_print_similarity_prob(matrix=matrix, user_id=user_id)


