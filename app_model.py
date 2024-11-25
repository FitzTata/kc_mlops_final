import os
from typing import Annotated, List

import gitlab
import mlflow
import uvicorn
from fastapi import Body, Depends, FastAPI
from pydantic import BaseModel, Field

from config import GITLAB_PROJECT_NAME, PROD_ALIAS, REGISTERED_MODEL_NAME

app = FastAPI()


# check that all env variables are present
for e in (
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "GITLAB_TOKEN",
    "GITLAB_TRIGGER_TOKEN",
):
    if e not in os.environ:
        raise ValueError(f"please set {e} env variable")

GITLAB_TOKEN = os.environ["GITLAB_TOKEN"]
GITLAB_TRIGGER_TOKEN = os.environ["GITLAB_TRIGGER_TOKEN"]


# define models for prediction endpoint
class PredictRequest(BaseModel):
    passwords: List[str] = Field(alias="Password")


class PredictResponse(BaseModel):
    prediction: List[float] = Field(alias="Times")


# define model cache and dependency for FastAPI
_model = None


def get_model():
    global _model
    if _model is None:
        _reload_model()
    return _model


@app.post("/predict")
def predict(
    data: Annotated[PredictRequest, Body()], model=Depends(get_model)
) -> PredictResponse:
    """Prediction endpoint"""
    prediction = model.predict(data.passwords)
    return PredictResponse(Times=prediction)


def _reload_model():
    """Loads latest registered model from mlflow with PROD alias"""
    global _model
    _model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@{PROD_ALIAS}")


@app.post("/reload-model")
def reload_model():
    """Model hot reload endpoint"""
    _reload_model()


# define model for trigger endpoint
class Trigger(BaseModel):
    data_url: str


@app.post("/trigger")
def trigger_pipeline(data: Annotated[Trigger, Body()]):
    """Trigger pipeline endpoint"""
    # get Gitlab API instance
    gl = gitlab.Gitlab(url="https://git.lab.karpov.courses", private_token=GITLAB_TOKEN)
    # load repo (project)
    project = gl.projects.get(GITLAB_PROJECT_NAME)

    # trigger pipeline on main branch via API
    project.trigger_pipeline(
        "main",
        GITLAB_TRIGGER_TOKEN,
        {
            "DATA_URL": data.data_url,
            "HOT_RELOAD_URL": "https://kc-mlops-project.onrender.com/reload-model",
        },
    )
    # do not wait for pipeline to finish and return


@app.get("/health")
def health():
    return "OK"


def main():
    uvicorn.run(app, host="0.0.0.0")


if __name__ == "__main__":
    main()
