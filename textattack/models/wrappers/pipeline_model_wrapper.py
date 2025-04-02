from .model_wrapper import ModelWrapper

class PipelineModelWrapper(ModelWrapper):
    def __init__(self, model):
        self.model = model

    def __call__(self, input_texts):
        ret = []
        for i in input_texts:
            pred = self.model(i)
            ret.append(pred)
        return ret