import torch
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import numpy as np
import time
import logging

from .model_wrapper import ModelWrapper

from toxic.core.bert_pytorch import BertForMultiLabelSequenceClassification, InputExample, convert_examples_to_features
from toxic.config import DEFAULT_MODEL_PATH, LABEL_LIST

logger = logging.getLogger()

class IBMBertModelWrapper(ModelWrapper):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        # 1. set appropriate parameters
        self.eval_batch_size = 64
        self.max_seq_length = 256
        self.do_lower_case = True


        self.device = torch.device("cpu")
        self.model.to(self.device)

        # 3. Set the layers to evaluation mode
        self.model.eval()

        logger.info('Loaded model')

    def _pre_process(self, input_texts):
        """Converts raw text into model-ready tensors."""
        start_time = time.time()

        # Convert input text to InputExample format
        test_examples = [InputExample(guid=i, text_a=text, labels=[]) for i, text in enumerate(input_texts)]

        # Convert examples to BERT-compatible input features
        test_features = convert_examples_to_features(test_examples, self.max_seq_length, self.tokenizer)

        # Convert to PyTorch tensors
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)

        # Create DataLoader for batch processing
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.eval_batch_size)

        # print(f"🔹 Preprocessing done in {time.time() - start_time:.2f} seconds.")
        return test_dataloader

    def _post_process(self, logits):
        """Converts model output logits into a dictionary of label probabilities."""
        output = [{LABEL_LIST[i]: prob for i, prob in enumerate(logit)} for logit in logits]
        return output

    def __call__(self, input_texts):
        """Runs inference on input text."""
        test_dataloader = self._pre_process(input_texts)

        all_logits = None

        for batch in test_dataloader:
            input_ids, input_mask, segment_ids = batch
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                logits = logits.sigmoid()  # Convert to probabilities

            if all_logits is None:
                all_logits = logits.detach().cpu().numpy()
            else:
                all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

        # Convert raw logits to human-readable output
        return self._post_process(all_logits)
