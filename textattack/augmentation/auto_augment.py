import random

from datasets import Dataset as h_Dataset
import transformers

import textattack
from textattack.augmentation.recipes import (
    CheckListAugmenter,
    CLAREAugmenter,
    EasyDataAugmenter,
    SynonymInsertionAugmenter,
    WordNetAugmenter,
)
from textattack.datasets import Dataset, HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper, ModelWrapper


class AutoAugmenter:
    def __init__(
        self,
        model="bert-base-uncased",
        train_dataset=None,
        eval_dataset=None,
        augmenters=None,
        training_args=None,
        split_of_train_dataset=100,
        split_of_eval_dataset=100,
        goal_eval_score=None,
        max_iteration=3,
    ):
        if augmenters is None:
            augmenters = [WordNetAugmenter, SynonymInsertionAugmenter, CheckListAugmenter]

        if isinstance(model, textattack.models.wrappers.ModelWrapper):
            self.model = model
        else:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                model
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(model)
            self.model = HuggingFaceModelWrapper(model, tokenizer)

        self.augmenters = augmenters

        if training_args:
            self.training_args = training_args
        else:
            self.training_args = textattack.TrainingArgs(
                num_epochs=2,
                num_clean_epochs=1,
                num_train_adv_examples=1000,
                learning_rate=5e-5,
                per_device_train_batch_size=8,
                gradient_accumulation_steps=4,
                log_to_tb=True,
            )

        if train_dataset:
            self.train_dataset = train_dataset
        else:
            self.train_dataset = HuggingFaceDataset("imdb", split="train")

        if eval_dataset:
            self.eval_dataset = eval_dataset
        else:
            self.eval_dataset = HuggingFaceDataset("imdb", split="eval")

        self.train_dataset.shard(split_of_train_dataset, 0)
        self.eval_dataset.shard(split_of_eval_dataset, 0)

        dict = train_dataset.dataset.to_dict()
        self.dict_texts = dict["text"]
        self.dict_labels = dict["label"]

        self.max_iteration = max_iteration
        self.goal_eval_score = goal_eval_score

    def greedy_train(self):

        previous_augmenters = []

        for _ in range(self.max_iteration):

            eval_scores = []

            for a in self.augmenters:

                augmenter_ls = previous_augmenters + [a]

                text_list = self.dict_texts
                label_list = self.dict_labels

                for i in range(len(self.dict_texts)):
                    text = self.dict_texts[i]
                    label = self.dict_labels[i]

                    potential_texts = []
                    for augmenter in augmenter_ls:
                        potential_texts.append(augmenter.augment(text)[0])
                    augmented_text = random.choice(potential_texts)
                    text_list.append(augmented_text)
                    label_list.append(label)

                augmented_dict = {"text": text_list, "label": label_list}
                augmented_dataset = h_Dataset.from_dict(augmented_dict)
                augmented_dataset = HuggingFaceDataset(
                    name_or_dataset=augmented_dataset
                )

                trainer = textattack.Trainer(
                    self.model,
                    "classification",
                    None,
                    augmented_dataset,
                    self.eval_dataset,
                    self.training_args,
                )

                eval_score = trainer.train()
                eval_scores.append(eval_score)

            max_score = max(eval_scores)
            index = eval_scores.index(max_score)
            previous_augmenters = previous_augmenters.append(self.augmenters[index])

        return previous_augmenters
