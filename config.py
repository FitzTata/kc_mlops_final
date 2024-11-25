"""Config variables"""

import os

PROD_ALIAS = "prod"
REGISTERED_MODEL_NAME = "a-davydov-project-model"
GITLAB_PROJECT_NAME = "aleksandr-davydov-wtf9954/course-project"

os.environ["MLFLOW_TRACKING_URI"] = "https://lab-mlflow.karpov.courses"
os.environ["AWS_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
