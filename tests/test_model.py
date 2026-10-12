
from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf


MODEL_PATH = Path("artifacts/training/model.h5")


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Trained model is not available",
)
def test_model_prediction_shape():
    model = tf.keras.models.load_model(MODEL_PATH)

    image = np.zeros((1, 224, 224, 3), dtype=np.float32)
    predictions = model.predict(image, verbose=0)

    assert predictions.shape == (1, 2)
    assert np.all(np.isfinite(predictions))
