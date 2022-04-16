from kfp.v2.dsl import component

# project=PROJECT_ID
# train_file_path = TRAINING_FILE_PATH
# validation_file_path = VALIDATION_FILE_PATH

@component(
    # this component builds the recommender model with BigQuery ML
    packages_to_install=["google-cloud-bigquery", "google-cloud-aiplatform==1.7.1", "pandas", "fsspec", "gcsfs","pyarrow","fastparquet"],
    base_image="python:3.8",
    output_component_file="data_split.yaml"
)
def train_valid_split(project:str,
                      train_file_path:str,
                      validation_file_path:str):
    
    from google.cloud import bigquery
    def create_query(phase):
        if phase == "TRAIN":
            return """SELECT * 
                        FROM `movielens.ratings` AS table
                        WHERE MOD(ABS(FARM_FINGERPRINT(TO_JSON_STRING(table))), 10) IN (1, 2, 3, 4)"""
        elif phase == "VALID":
            return """SELECT *
                        FROM `movielens.ratings` AS table
                        WHERE MOD(ABS(FARM_FINGERPRINT(TO_JSON_STRING(table))), 10) IN (8)"""

    def get_file_path(phase):
        if phase == "TRAIN":
            return train_file_path
        return validation_file_path

    client = bigquery.Client(project=project)
    for phase in ["TRAIN", "VALID"]:  # TODO: add test dataset later, export to a bq table
        # 1. Create query string
        query_string = create_query(phase)

        # 2. Load results into DataFrame
        df = client.query(query_string).to_dataframe()

        # 3. Write DataFrame to CSV
        df.to_csv(get_file_path(phase), index_label=False, index=False)
        print("Wrote {} to {}".format(phase, get_file_path(phase)))
