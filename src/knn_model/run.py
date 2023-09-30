from pathlib import Path
from src.knn_model.dataset import get_train_test_data, normalize_user_id
from src.knn_model.model import get_matrix, add_new_empty_user, add_data_to_user, predict_real_user_id




if __name__ == "__main__":
  train_df, _ = get_train_test_data(data_path=Path("dataset/Keystrokes/files"), num_of_users=1000)
  train_df = normalize_user_id(train_df)

  matrix = get_matrix(train_df)
  matrix = add_new_empty_user(matrix=matrix)
  add_data_to_user(matrix=matrix, user_id=len(matrix) - 1)
  predict_real_user_id(matrix)


