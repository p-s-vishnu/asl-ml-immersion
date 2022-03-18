"""Kubeflow Covertype Pipeline."""

import os
from google.cloud.aiplatform import hyperparameter_tuning as hpt
from google_cloud_pipeline_components.aiplatform import (
    EndpointCreateOp,
    ModelDeployOp,
    ModelUploadOp,
)
from google_cloud_pipeline_components.experimental import (
    hyperparameter_tuning_job,
)
from google_cloud_pipeline_components.experimental.custom_job import (
    CustomTrainingJobOp,
)
from kfp.v2 import dsl

PIPELINE_ROOT = os.getenv("PIPELINE_ROOT")
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")

TRAINING_CONTAINER_IMAGE_URI = os.getenv("TRAINING_CONTAINER_IMAGE_URI")
SERVING_CONTAINER_IMAGE_URI = os.getenv("SERVING_CONTAINER_IMAGE_URI")
SERVING_MACHINE_TYPE = os.getenv("SERVING_MACHINE_TYPE", "n1-standard-16")

TRAINING_FILE_PATH = os.getenv("TRAINING_FILE_PATH")
VALIDATION_FILE_PATH = os.getenv("VALIDATION_FILE_PATH")

# MAX_TRIAL_COUNT = int(os.getenv("MAX_TRIAL_COUNT", "5"))
# PARALLEL_TRIAL_COUNT = int(os.getenv("PARALLEL_TRIAL_COUNT", "5"))
# THRESHOLD = float(os.getenv("THRESHOLD", "0.6"))

PIPELINE_NAME = os.getenv("PIPELINE_NAME", "covertype")
BASE_OUTPUT_DIR = os.getenv("BASE_OUTPUT_DIR", PIPELINE_ROOT)
MODEL_DISPLAY_NAME = os.getenv("MODEL_DISPLAY_NAME", PIPELINE_NAME)


@dsl.pipeline(
    name=f"forest-{PIPELINE_NAME}-kfp-pipeline",
    description="Kubeflow pipeline that Trains, and deploys on Vertex",
    pipeline_root=PIPELINE_ROOT,
)
def create_pipeline():

    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": "n1-standard-4",
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": TRAINING_CONTAINER_IMAGE_URI,
                "args": [
                    f"--training_dataset_path={TRAINING_FILE_PATH}",
                    f"--validation_dataset_path={VALIDATION_FILE_PATH}",
                    "--alpha=1.0e-1",
                    "--max_iter=1"
                ],
            },
        }
    ]

    training_task = CustomTrainingJobOp(
        project=PROJECT_ID,
        location=REGION,
        display_name=f"{PIPELINE_NAME}-kfp-training-job",
        worker_pool_specs=worker_pool_specs,
        base_output_directory=BASE_OUTPUT_DIR,
    )

    model_upload_task = ModelUploadOp(
        project=PROJECT_ID,
        display_name=f"{PIPELINE_NAME}-kfp-model-upload-job",
        artifact_uri=f"{BASE_OUTPUT_DIR}/model",
        serving_container_image_uri=SERVING_CONTAINER_IMAGE_URI,
    )
    model_upload_task.after(training_task)

    endpoint_create_task = EndpointCreateOp(
        project=PROJECT_ID,
        display_name=f"{PIPELINE_NAME}-kfp-create-endpoint-job",
    )
    endpoint_create_task.after(model_upload_task)

    model_deploy_op = ModelDeployOp(  # pylint: disable=unused-variable
        model=model_upload_task.outputs["model"],
        endpoint=endpoint_create_task.outputs["endpoint"],
        deployed_model_display_name=MODEL_DISPLAY_NAME,
        dedicated_resources_machine_type=SERVING_MACHINE_TYPE,
        dedicated_resources_min_replica_count=1,
        dedicated_resources_max_replica_count=1,
    )
