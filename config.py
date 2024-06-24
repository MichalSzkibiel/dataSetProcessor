import os

# Workspace where everything will be saved. Defaults to "."
workspace = "."

# Pass here your credentials to https://dataspace.copernicus.eu/
copernicus_token_data = {
    "client_id": "cdse-public",
    "username": "USER_NAME",
    "password": "PASSWORD",
    "grant_type": "password"
}

english = {
    'model_heading': 'Learning of model',
    'learning_parameters': 'Parameters for dataset preparation',
    'parameter_name': 'Name',
    'parameter_value': 'Value',
    True: 'Yes',
    False: 'No',
    'features_count': 'Number of features',
    'features_params': 'Number of arguments',
    'normalization': 'Standard Normalization',
    'pca': 'Applied PCA decomposition',
    'pca_explained_threshold': 'PCA decomposition explain threshold',
    'pca_explained_final': 'PCA decomposition explains',
    'pca_arguments': 'Remaining components after PCA decomposition',
    'test_ratio': 'Percent of test dataset',
    'training_count': 'Number of features in training dataset',
    'test_count': 'Number of features in test dataset',
    'test\\training': 'Test\\Training',
    'precision': 'Precision',
    'recall': 'Recall',
    'accuracy': 'Accuracy',
    'f1': 'F1 Score',
    'f1_mean': 'Mean F1 Score',
    'model_type': 'Type of Model',
    'random_forest': 'Random Forest',
    'n_estimators': 'Number of Estimators',
    'fitting_time': 'Fitting time',
    'output_activation': 'Activation function of output layer',
    'optimizer': 'Optimizer',
    'loss': 'Loss function',
    'batch_size': 'Batch size',
    'epochs': 'Epochs',
    'loss_diff': 'Minimal difference in loss between iterations',
    'epochs_count': 'Number of passed epochs',
    'model_parameters': 'Model',
    'hidden_layers': 'Hidden layers count',
    'hidden_layer_size': 'Size of hidden layers',
    'confusion_matrix': 'Confusion matrix and metrics',
    'dataset_name': 'Name of the dataset',
    'dataset_heading': 'Dataset',
    'dataset_parameters': 'Parameters',
    'bbox': 'Bounding box',
    'sentinel2_name': 'Name of Sentinel2 image',
    'sentinel2_date': 'Date of Sentinel2 image',
    'window_size': 'Size of processing window',
    'ignore_off_scope': 'If window of pixel exceeds image scope, it will be ignored',
    'is_conv': 'Dataset has shape for convolutional neural network',
    'rotations_and_reflections': 'Data was augmented by rotations and reflections',
    'labels_count': 'Count of features with given labels',
    'class': 'Label',
    'count': 'Count',
    'first_conv_filters': 'Filters in first convolutional layer',
    'second_conv_filters': 'Filters in second convolutional layer',
    'third_conv_filters': 'Filters in third convolutional layer',
    'multilearn_heading': 'Comparison of models',
    'multilearn_parameters': 'Datasets',
    'train_dataset': 'Train dataset name',
    'test_dataset': 'Test dataset name',
    'models_performance': 'Performance of models',
    'model_name': 'Name of model',
    'train_accuracy': 'Accuracy on train dataset',
    'train_mean_f1': 'Mean F1 score on train dataset',
    'train_kappa': 'Kappa coefficient on train dataset',
    'test_accuracy': 'Accuracy on test dataset',
    'test_kappa': 'Kappa coefficient on test dataset',
    'nearest_centroid': 'Nearest Centroid',
    'svm': 'Support Vector Machine',
    'ml': 'Maximum likelihood',
    'xgboost': 'XGBoost',
    'max_depth': 'Maximum depth',
    'kappa': 'Kappa coeficient',
    'sum': 'sum'
}

language = english


def create_workspace():
    if not os.path.exists(workspace):
        os.mkdir(workspace)
    for el in [
        "classImages",
        "cloud_masks",
        "compositions",
        "datasets",
        "images",
        "models",
        "parcels",
        "sentinel2images",
        "temp",
        "test_datasets"
    ]:
        if not os.path.exists(os.path.join(workspace, el)):
            os.mkdir(os.path.join(workspace, el))
    reports_path = os.path.join(workspace, "reports")
    if not os.path.exists(reports_path):
        os.mkdir(reports_path)
    for el in ["classifying", "datasets", "images", "learning"]:
        if not os.path.exists(os.path.join(reports_path, el)):
            os.mkdir(os.path.join(reports_path, el))
