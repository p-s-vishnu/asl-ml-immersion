from kfp.v2.dsl import component

@component(
    # this component builds the recommender model with BigQuery ML
    packages_to_install=["google-cloud-bigquery","tensorflow", "tensorflow_datasets", "pandas", "fsspec", "gcsfs","pyarrow","fastparquet"],
    base_image="python:3.9",
    output_component_file="data_prep.yaml"
)
def data_prep(training_file_path:str,
              validation_file_path:str,
              training_clean_file_path:str,
              validation_clean_file_path:str):
    import pandas as pd
    import numpy as np

    df_train = pd.read_csv(training_file_path)
    df_val = pd.read_csv(validation_file_path)

    movies_ids = list(set(list(df_train.movieId.unique()) + list(df_val.movieId.unique())))
    users_ids = list(set(list(df_train.userId.unique()) + list(df_val.userId.unique())))

    dict_movies = dict(zip(movies_ids, range(len(movies_ids)) ))
    dict_users = dict(zip(users_ids, range(len(users_ids)) ))

    df_train["movieId"] = df_train["movieId"].map(dict_movies)
    df_val["movieId"] = df_val["movieId"].map(dict_movies)

    df_train["userId"] = df_train["userId"].map(dict_users)
    df_val["userId"] = df_val["userId"].map(dict_users)

    col = ["userId", "movieId", "rating"]
    df_train[col] = df_train[col].astype(np.float32)
    df_val[col] = df_val[col].astype(np.float32)

    # save to bucket
    df_train.to_csv(training_clean_file_path, index=False)
    df_val.to_csv(validation_clean_file_path, index=False)
    
