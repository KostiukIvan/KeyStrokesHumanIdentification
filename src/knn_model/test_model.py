from pathlib import Path
from src.knn_model.dataset import get_train_test_data, normalize_user_id
from src.knn_model.model import get_matrix, add_new_empty_user, add_data_to_user, get_id_of_user_based_on_sample, _get_mean_values_of_matrix


def test_knn_model():
    train_df, test_df = get_train_test_data(data_path=Path("dataset/Keystrokes/files"), num_of_users=1000, test_ratio=0.3)

    train_matrix = get_matrix(train_df)
    test_matrix = get_matrix(test_df)

    train_matrix = _get_mean_values_of_matrix(train_matrix)
    test_matrix = _get_mean_values_of_matrix(test_matrix)

    success_rate = 0
    total_users = 0
    for user_id, sample in enumerate(test_matrix):
        pred_id, _ = get_id_of_user_based_on_sample(train_matrix, [sample])
        total_users += 1
        success_rate += (pred_id == user_id)

    assert success_rate / total_users > 0.5, f"success rate is {success_rate / total_users}"


