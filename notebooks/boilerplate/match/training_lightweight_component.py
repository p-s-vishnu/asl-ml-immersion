
"""Lightweight component training function."""
from kfp.v2.dsl import component
import os

DATASET = os.environ["DATASET"]

@component(
    # this component builds the recommender model
    packages_to_install=["google-cloud-bigquery", "sklearn","tensorflow", "tensorflow_datasets", "pandas", "fsspec", "gcsfs","pyarrow","fastparquet"],
    base_image="python:3.9",
    output_component_file="train_and_fit.yaml"
)
def train(training_clean_file_path: str,
          validation_clean_file_path: str,
          model_dir: str):    
    # TODO: Move to train.py script
    import pandas as pd
    import numpy as np
    import logging
    from typing import Dict, Text
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras import optimizers as opt
    from tensorflow.keras.layers import Embedding, multiply, concatenate, Flatten, Input, Dense

    logging.info("Reading train data...")    
    df_train = pd.read_csv(training_clean_file_path)
    logging.info("Reading valid data...")
    df_val = pd.read_csv(validation_clean_file_path)

    num_unique_users=len(set(list(df_train.userId.unique()) + list(df_val.userId.unique())))
    num_unique_movies=len(set(list(df_train.movieId.unique()) + list(df_val.movieId.unique())))

    users_input = Input(shape=(1,), name="users_input")
    users_embedding = Embedding(num_unique_users + 1, 50, name="users_embeddings")(users_input)
    users_bias = Embedding(num_unique_users + 1, 1, name="users_bias")(users_input)

    movies_input = Input(shape=(1,), name="movies_input")
    movies_embedding = Embedding(num_unique_movies + 1, 50, name="movies_embedding")(movies_input)
    movies_bias = Embedding(num_unique_movies + 1, 1, name="movies_bias")(movies_input)

    dot_product_users_movies = multiply([users_embedding, movies_embedding])
    input_terms = dot_product_users_movies + users_bias + movies_bias
    input_terms = Flatten(name="fl_inputs")(input_terms)
    output = Dense(1, activation="relu", name="output")(input_terms)
    model = Model(inputs=[users_input, movies_input], outputs=output)

    opt_adam = opt.Adam(lr = 0.005)
    model.compile(optimizer=opt_adam, loss= ['mse'], metrics=['mean_absolute_error'])

    model.fit(x=[df_train.userId, df_train.movieId], 
              y=df_train.rating, 
              batch_size=512, 
              epochs=1, 
              verbose=1, 
              validation_data=([df_val.userId, df_val.movieId], df_val.rating))

    # save model
    tf.saved_model.save(model, model_dir)


@component(
    base_image="python:3.8",
    packages_to_install=["google-cloud-aiplatform", "joblib", "sklearn", "google-cloud-bigquery"],
    output_component_file= f"{DATASET}_kfp_deploy.yaml",
)
def deploy(
    project: str,
    location: str,
    serving_container_uri: str,
    display_name:str,
    artifact_uri:str, 
):
    from google.cloud import aiplatform
    import os
    
    aiplatform.init(project=project)
    
    deployed_model = aiplatform.Model.upload(
        display_name= display_name,
        artifact_uri = artifact_uri,
        serving_container_image_uri= serving_container_uri
    )
    endpoint = deployed_model.deploy(
        traffic_split={"0": 100},
        machine_type="n1-standard-4"
    )
