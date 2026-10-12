
import dagshub
import tensorflow as tf
import mlflow
from mlflow.tracking import MlflowClient
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.comman import save_json
from pathlib import Path


dagshub.init(
    repo_owner="skbad911",
    repo_name="chicken-disease-classification",
    mlflow=True
)

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self._valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self._valid_generator)

    def save_score(self):
        scores = {
            "loss": float(self.score[0]),
            "accuracy": float(self.score[1])
        }

        save_json(path=Path("scores.json"), data=scores)

        with mlflow.start_run() as run:
            # Log evaluation metrics
            mlflow.log_metrics(scores)

            # Log configuration parameters
            params = self.config.all_params
            if params:
                mlflow.log_params({
                    key: str(value)
                    for key, value in params.items()
                    if isinstance(value, (str, int, float, bool))
                })

            # Log the trained TensorFlow model
            mlflow.log_artifact(
                str(self.config.path_of_model),
                artifact_path="model"
            )

            # Verify artifacts were recorded
            client = MlflowClient()
            artifacts = client.list_artifacts(run.info.run_id)

            print("MLflow run ID:", run.info.run_id)
            print("MLflow artifact URI:", run.info.artifact_uri)
            print("MLflow artifacts:", [item.path for item in artifacts])
