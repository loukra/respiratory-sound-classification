import numpy as np


class MyPredictor(object):
    def __init__(self, model, preprocessor):
        self._model = model
        self._preprocessor = preprocessor

    def predict(self, instances, **kwargs):
        inputs = np.asarray(instances)
        preprocessed_inputs = self._preprocessor.preprocess(inputs)
        outputs = self._model.predict(preprocessed_inputs)
        if kwargs.get("chunks"):
            return outputs.tolist()
        return np.mean(outputs)
