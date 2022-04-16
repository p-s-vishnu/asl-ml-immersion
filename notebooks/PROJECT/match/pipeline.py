

import os
from kfp import dsl
# change the below imports if you change the module name
from data import data_prep
from split import train_valid_split
from training_lightweight_component import train, deploy


DATASET = os.getenv("DATASET")
DATA_ROOT = os.getenv("DATA_ROOT")
ARTIFACT_STORE = os.getenv("ARTIFACT_STORE")
PIPELINE_ROOT = os.getenv("PIPELINE_ROOT")
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")

TRAINING_CONTAINER_IMAGE_URI = os.getenv("TRAINING_CONTAINER_IMAGE_URI")
SERVING_CONTAINER_IMAGE_URI = os.getenv("SERVING_CONTAINER_IMAGE_URI")

TRAINING_FILE_PATH = os.getenv("TRAINING_FILE_PATH")
VALIDATION_FILE_PATH = os.getenv("VALIDATION_FILE_PATH")
TRAINING_CLEAN_FILE_PATH = f"{DATA_ROOT}/training_clean/dataset.csv"
VALIDATION_CLEAN_FILE_PATH= f"{DATA_ROOT}/validation_clean/dataset.csv"

MAX_TRIAL_COUNT = int(os.getenv("MAX_TRIAL_COUNT", "5"))
PARALLEL_TRIAL_COUNT = int(os.getenv("PARALLEL_TRIAL_COUNT", "5"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.6"))

MODEL_DIR = f"{ARTIFACT_STORE}/model"

@dsl.pipeline(
    name=f"{DATASET}-kfp-pipeline",
    description=f"The pipeline training and deploying the {DATASET} model",
    pipeline_root=PIPELINE_ROOT,
)
def pipeline(
    training_container_uri: str = TRAINING_CONTAINER_IMAGE_URI,
    serving_container_uri: str = SERVING_CONTAINER_IMAGE_URI,
    training_file_path: str = TRAINING_FILE_PATH,
    validation_file_path: str = VALIDATION_FILE_PATH,
    training_clean_file_path: str = TRAINING_CLEAN_FILE_PATH,
    validation_clean_file_path: str = VALIDATION_CLEAN_FILE_PATH,
    model_dir: str = MODEL_DIR,
    accuracy_deployment_threshold: float = THRESHOLD,
    max_trial_count: int = MAX_TRIAL_COUNT,
    parallel_trial_count: int = PARALLEL_TRIAL_COUNT,
    pipeline_root: str = PIPELINE_ROOT,
):
    train_valid_op = train_valid_split(project=PROJECT_ID,
                                       train_file_path = TRAINING_FILE_PATH,
                                       validation_file_path = VALIDATION_FILE_PATH)
    
    data_prep_op = data_prep(training_file_path=training_file_path,
                             validation_file_path=validation_file_path,
                             training_clean_file_path= training_clean_file_path,
                             validation_clean_file_path= validation_clean_file_path)
    data_prep_op.after(train_valid_op)
    
    train_op   = train(training_clean_file_path = training_clean_file_path,
                       validation_clean_file_path= validation_clean_file_path,
                       model_dir = model_dir)
    train_op.after(data_prep_op)
    
    deploy_op = deploy(
        project=PROJECT_ID,
        location=REGION,
        serving_container_uri=serving_container_uri,
        display_name='movie-recommender-keras',
        artifact_uri= model_dir
    )
    deploy_op.after(train_op)
