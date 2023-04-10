"""
Trainer Class
=============
"""

import collections
import json
import logging
import math
import os

import scipy
import torch
import tqdm
import transformers

import textattack
from textattack.shared.utils import logger

from .attack import Attack
from .attack_args import AttackArgs
from .attack_results import MaximizedAttackResult, SuccessfulAttackResult
from .attacker import Attacker
from .model_args import HUGGINGFACE_MODELS
from .models.helpers import LSTMForClassification, WordCNNForClassification
from .models.wrappers import ModelWrapper
from .training_args import CommandLineTrainingArgs, TrainingArgs


class Trainer:
    """Trainer is training and eval loop for adversarial training.

    It is designed to work with PyTorch and Transformers models.

    Args:
        model_wrapper (:class:`~textattack.models.wrappers.ModelWrapper`):
            Model wrapper containing both the model and the tokenizer.
        task_type (:obj:`str`, `optional`, defaults to :obj:`"classification"`):
            The task that the model is trained to perform.
            Currently, :class:`~textattack.Trainer` supports two tasks: (1) :obj:`"classification"`, (2) :obj:`"regression"`.
        attack (:class:`~textattack.Attack`):
            :class:`~textattack.Attack` used to generate adversarial examples for training.
        train_dataset (:class:`~textattack.datasets.Dataset`):
            Dataset for training.
        eval_dataset (:class:`~textattack.datasets.Dataset`):
            Dataset for evaluation
        training_args (:class:`~textattack.TrainingArgs`):
            Arguments for training.

    Example::

        >>> import textattack
        >>> import transformers

        >>> model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        >>> tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

        >>> # We only use DeepWordBugGao2018 to demonstration purposes.
        >>> attack = textattack.attack_recipes.DeepWordBugGao2018.build(model_wrapper)
        >>> train_dataset = textattack.datasets.HuggingFaceDataset("imdb", split="train")
        >>> eval_dataset = textattack.datasets.HuggingFaceDataset("imdb", split="test")

        >>> # Train for 3 epochs with 1 initial clean epochs, 1000 adversarial examples per epoch, learning rate of 5e-5, and effective batch size of 32 (8x4).
        >>> training_args = textattack.TrainingArgs(
        ...     num_epochs=3,
        ...     num_clean_epochs=1,
        ...     num_train_adv_examples=1000,
        ...     learning_rate=5e-5,
        ...     per_device_train_batch_size=8,
        ...     gradient_accumulation_steps=4,
        ...     log_to_tb=True,
        ... )

        >>> trainer = textattack.Trainer(
        ...     model_wrapper,
        ...     "classification",
        ...     attack,
        ...     train_dataset,
        ...     eval_dataset,
        ...     training_args
        ... )
        >>> trainer.train()

    .. note::
        When using :class:`~textattack.Trainer` with `parallel=True` in :class:`~textattack.TrainingArgs`,
        make sure to protect the “entry point” of the program by using :obj:`if __name__ == '__main__':`.
        If not, each worker process used for generating adversarial examples will execute the training code again.
    """

    def __init__(
        self,
        model_wrapper,
        task_type="classification",
        attack=None,
        train_dataset=None,
        eval_dataset=None,
        training_args=None,
    ):
        assert isinstance(
            model_wrapper, ModelWrapper
        ), f"`model_wrapper` must be of type `textattack.models.wrappers.ModelWrapper`, but got type `{type(model_wrapper)}`."

        # TODO: Support seq2seq training
        assert task_type in {
            "classification",
            "regression",
        }, '`task_type` must either be "classification" or "regression"'

        if attack:
            assert isinstance(
                attack, Attack
            ), f"`attack` argument must be of type `textattack.Attack`, but got type of `{type(attack)}`."

            if id(model_wrapper) != id(attack.goal_function.model):
                logger.warn(
                    "`model_wrapper` and the victim model of `attack` are not the same model."
                )

        if train_dataset:
            assert isinstance(
                train_dataset, textattack.datasets.Dataset
            ), f"`train_dataset` must be of type `textattack.datasets.Dataset`, but got type `{type(train_dataset)}`."

        if eval_dataset:
            assert isinstance(
                eval_dataset, textattack.datasets.Dataset
            ), f"`eval_dataset` must be of type `textattack.datasets.Dataset`, but got type `{type(eval_dataset)}`."

        if training_args:
            assert isinstance(
                training_args, TrainingArgs
            ), f"`training_args` must be of type `textattack.TrainingArgs`, but got type `{type(training_args)}`."
        else:
            training_args = TrainingArgs()

        if not hasattr(model_wrapper, "model"):
            raise ValueError("Cannot detect `model` in `model_wrapper`")
        else:
            assert isinstance(
                model_wrapper.model, torch.nn.Module
            ), f"`model` in `model_wrapper` must be of type `torch.nn.Module`, but got type `{type(model_wrapper.model)}`."

        if not hasattr(model_wrapper, "tokenizer"):
            raise ValueError("Cannot detect `tokenizer` in `model_wrapper`")

        self.model_wrapper = model_wrapper
        self.task_type = task_type
        self.attack = attack
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.training_args = training_args

        self._metric_name = (
            "pearson_correlation" if self.task_type == "regression" else "accuracy"
        )
        if self.task_type == "regression":
            self.loss_fct = torch.nn.MSELoss(reduction="none")
        else:
            self.loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        self._global_step = 0

    def _generate_adversarial_examples(self, epoch):
        """Generate adversarial examples using attacker."""
        assert (
            self.attack is not None
        ), "`attack` is `None` but attempting to generate adversarial examples."
        base_file_name = f"attack-train-{epoch}"
        log_file_name = os.path.join(self.training_args.output_dir, base_file_name)
        logger.info("Attacking model to generate new adversarial training set...")

        if isinstance(self.training_args.num_train_adv_examples, float):
            num_train_adv_examples = math.ceil(
                len(self.train_dataset) * self.training_args.num_train_adv_examples
            )
        else:
            num_train_adv_examples = self.training_args.num_train_adv_examples

        # Use Different AttackArgs based on num_train_adv_examples value.
        # If num_train_adv_examples >= 0 , num_train_adv_examples is
        # set as number of successful examples.
        # If num_train_adv_examples == -1 , num_examples is set to -1 to
        # generate example for all of training data.
        if num_train_adv_examples >= 0:
            attack_args = AttackArgs(
                num_successful_examples=num_train_adv_examples,
                num_examples_offset=0,
                query_budget=self.training_args.query_budget_train,
                shuffle=True,
                parallel=self.training_args.parallel,
                num_workers_per_device=self.training_args.attack_num_workers_per_device,
                disable_stdout=True,
                silent=True,
                log_to_txt=log_file_name + ".txt",
                log_to_csv=log_file_name + ".csv",
            )
        elif num_train_adv_examples == -1:
            # set num_examples when num_train_adv_examples = -1
            attack_args = AttackArgs(
                num_examples=num_train_adv_examples,
                num_examples_offset=0,
                query_budget=self.training_args.query_budget_train,
                shuffle=True,
                parallel=self.training_args.parallel,
                num_workers_per_device=self.training_args.attack_num_workers_per_device,
                disable_stdout=True,
                silent=True,
                log_to_txt=log_file_name + ".txt",
                log_to_csv=log_file_name + ".csv",
            )
        else:
            assert False, "num_train_adv_examples is negative and not equal to -1."

        attacker = Attacker(self.attack, self.train_dataset, attack_args=attack_args)
        results = attacker.attack_dataset()

        attack_types = collections.Counter(r.__class__.__name__ for r in results)
        total_attacks = (
            attack_types["SuccessfulAttackResult"] + attack_types["FailedAttackResult"]
        )
        success_rate = attack_types["SuccessfulAttackResult"] / total_attacks * 100
        logger.info(f"Total number of attack results: {len(results)}")
        logger.info(
            f"Attack success rate: {success_rate:.2f}% [{attack_types['SuccessfulAttackResult']} / {total_attacks}]"
        )
        # TODO: This will produce a bug if we need to manipulate ground truth output.

        # To Fix Issue #498 , We need to add the Non Output columns in one tuple to represent input columns
        # Since adversarial_example won't be an input to the model , we will have to remove it from the input
        # dictionary in collate_fn
        adversarial_examples = [
            (
                tuple(r.perturbed_result.attacked_text._text_input.values())
                + ("adversarial_example",),
                r.perturbed_result.ground_truth_output,
            )
            for r in results
            if isinstance(r, (SuccessfulAttackResult, MaximizedAttackResult))
        ]

        # Name for column indicating if an example is adversarial is set as "_example_type".
        adversarial_dataset = textattack.datasets.Dataset(
            adversarial_examples,
            input_columns=self.train_dataset.input_columns + ("_example_type",),
            label_map=self.train_dataset.label_map,
            label_names=self.train_dataset.label_names,
            output_scale_factor=self.train_dataset.output_scale_factor,
            shuffle=False,
        )
        return adversarial_dataset

    def _print_training_args(
        self, total_training_steps, train_batch_size, num_clean_epochs
    ):
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num epochs = {self.training_args.num_epochs}")
        logger.info(f"  Num clean epochs = {num_clean_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {self.training_args.per_device_train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {train_batch_size * self.training_args.gradient_accumulation_steps}"
        )
        logger.info(
            f"  Gradient accumulation steps = {self.training_args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {total_training_steps}")

    def _save_model_checkpoint(
        self, model, tokenizer, step=None, epoch=None, best=False, last=False
    ):
        # Save model checkpoint
        if step:
            dir_name = f"checkpoint-step-{step}"
        if epoch:
            dir_name = f"checkpoint-epoch-{epoch}"
        if best:
            dir_name = "best_model"
        if last:
            dir_name = "last_model"

        output_dir = os.path.join(self.training_args.output_dir, dir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        if isinstance(model, (WordCNNForClassification, LSTMForClassification)):
            model.save_pretrained(output_dir)
        elif isinstance(model, transformers.PreTrainedModel):
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        else:
            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(
                state_dict,
                os.path.join(output_dir, "pytorch_model.bin"),
            )

    def _tb_log(self, log, step):
        if not hasattr(self, "_tb_writer"):
            from torch.utils.tensorboard import SummaryWriter

            self._tb_writer = SummaryWriter(self.training_args.tb_log_dir)
            self._tb_writer.add_hparams(self.training_args.__dict__, {})
            self._tb_writer.flush()

        for key in log:
            self._tb_writer.add_scalar(key, log[key], step)

    def _wandb_log(self, log, step):
        if not hasattr(self, "_wandb_init"):
            global wandb
            import wandb

            self._wandb_init = True
            wandb.init(
                project=self.training_args.wandb_project,
                config=self.training_args.__dict__,
            )

        wandb.log(log, step=step)

    def get_optimizer_and_scheduler(self, model, num_training_steps):
        """Returns optimizer and scheduler to use for training. If you are
        overriding this method and do not want to use a scheduler, simply
        return :obj:`None` for scheduler.

        Args:
            model (:obj:`torch.nn.Module`):
                Model to be trained. Pass its parameters to optimizer for training.
            num_training_steps (:obj:`int`):
                Number of total training steps.
        Returns:
            Tuple of optimizer and scheduler :obj:`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]`
        """
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        if isinstance(model, transformers.PreTrainedModel):
            # Reference https://huggingface.co/transformers/training.html
            param_optimizer = list(model.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.training_args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer = transformers.optimization.AdamW(
                optimizer_grouped_parameters, lr=self.training_args.learning_rate
            )
            if isinstance(self.training_args.num_warmup_steps, float):
                num_warmup_steps = math.ceil(
                    self.training_args.num_warmup_steps * num_training_steps
                )
            else:
                num_warmup_steps = self.training_args.num_warmup_steps

            scheduler = transformers.optimization.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda x: x.requires_grad, model.parameters()),
                lr=self.training_args.learning_rate,
            )
            scheduler = None

        return optimizer, scheduler

    def get_train_dataloader(self, dataset, adv_dataset, batch_size):
        """Returns the :obj:`torch.utils.data.DataLoader` for training.

        Args:
            dataset (:class:`~textattack.datasets.Dataset`):
                Original training dataset.
            adv_dataset (:class:`~textattack.datasets.Dataset`):
                Adversarial examples generated from the original training dataset. :obj:`None` if no adversarial attack takes place.
            batch_size (:obj:`int`):
                Batch size for training.
        Returns:
            :obj:`torch.utils.data.DataLoader`
        """

        # TODO: Add pairing option where we can pair original examples with adversarial examples.
        # Helper functions for collating data
        def collate_fn(data):
            input_texts = []
            targets = []
            is_adv_sample = []
            for item in data:
                if "_example_type" in item[0].keys():
                    # Get example type value from OrderedDict and remove it

                    adv = item[0].pop("_example_type")

                    # with _example_type removed from item[0] OrderedDict
                    # all other keys should be part of input
                    _input, label = item
                    if adv != "adversarial_example":
                        raise ValueError(
                            "`item` has length of 3 but last element is not for marking if the item is an `adversarial example`."
                        )
                    else:
                        is_adv_sample.append(True)
                else:
                    # else `len(item)` is 2.
                    _input, label = item
                    is_adv_sample.append(False)

                if isinstance(_input, collections.OrderedDict):
                    _input = tuple(_input.values())
                else:
                    _input = tuple(_input)

                if len(_input) == 1:
                    _input = _input[0]
                input_texts.append(_input)
                targets.append(label)

            return input_texts, torch.tensor(targets), torch.tensor(is_adv_sample)

        if adv_dataset:
            dataset = torch.utils.data.ConcatDataset([dataset, adv_dataset])

        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        return train_dataloader

    def get_eval_dataloader(self, dataset, batch_size):
        """Returns the :obj:`torch.utils.data.DataLoader` for evaluation.

        Args:
            dataset (:class:`~textattack.datasets.Dataset`):
                Dataset to use for evaluation.
            batch_size (:obj:`int`):
                Batch size for evaluation.
        Returns:
            :obj:`torch.utils.data.DataLoader`
        """

        # Helper functions for collating data
        def collate_fn(data):
            input_texts = []
            targets = []
            for _input, label in data:
                if isinstance(_input, collections.OrderedDict):
                    _input = tuple(_input.values())
                else:
                    _input = tuple(_input)

                if len(_input) == 1:
                    _input = _input[0]
                input_texts.append(_input)
                targets.append(label)
            return input_texts, torch.tensor(targets)

        eval_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        return eval_dataloader

    def training_step(self, model, tokenizer, batch):
        """Perform a single training step on a batch of inputs.

        Args:
            model (:obj:`torch.nn.Module`):
                Model to train.
            tokenizer:
                Tokenizer used to tokenize input text.
            batch (:obj:`tuple[list[str], torch.Tensor, torch.Tensor]`):
                By default, this will be a tuple of input texts, targets, and boolean tensor indicating if the sample is an adversarial example.

                .. note::
                    If you override the :meth:`get_train_dataloader` method, then shape/type of :obj:`batch` will depend on how you created your batch.

        Returns:
            :obj:`tuple[torch.Tensor, torch.Tensor, torch.Tensor]` where

            - **loss**: :obj:`torch.FloatTensor` of shape 1 containing the loss.
            - **preds**: :obj:`torch.FloatTensor` of model's prediction for the batch.
            - **targets**: :obj:`torch.Tensor` of model's targets (e.g. labels, target values).
        """

        input_texts, targets, is_adv_sample = batch
        _targets = targets
        targets = targets.to(textattack.shared.utils.device)

        if isinstance(model, transformers.PreTrainedModel) or (
            isinstance(model, torch.nn.DataParallel)
            and isinstance(model.module, transformers.PreTrainedModel)
        ):
            input_ids = tokenizer(
                input_texts,
                padding="max_length",
                return_tensors="pt",
                truncation=True,
            )
            input_ids.to(textattack.shared.utils.device)
            logits = model(**input_ids)[0]
        else:
            input_ids = tokenizer(input_texts)
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)
            input_ids = input_ids.to(textattack.shared.utils.device)
            logits = model(input_ids)

        if self.task_type == "regression":
            loss = self.loss_fct(logits.squeeze(), targets.squeeze())
            preds = logits
        else:
            loss = self.loss_fct(logits, targets)
            preds = logits.argmax(dim=-1)

        sample_weights = torch.ones(
            is_adv_sample.size(), device=textattack.shared.utils.device
        )
        sample_weights[is_adv_sample] *= self.training_args.alpha
        loss = loss * sample_weights
        loss = torch.mean(loss)
        preds = preds.cpu()

        return loss, preds, _targets

    def evaluate_step(self, model, tokenizer, batch):
        """Perform a single evaluation step on a batch of inputs.

        Args:
            model (:obj:`torch.nn.Module`):
                Model to train.
            tokenizer:
                Tokenizer used to tokenize input text.
            batch (:obj:`tuple[list[str], torch.Tensor]`):
                By default, this will be a tuple of input texts and target tensors.

                .. note::
                    If you override the :meth:`get_eval_dataloader` method, then shape/type of :obj:`batch` will depend on how you created your batch.

        Returns:
            :obj:`tuple[torch.Tensor, torch.Tensor]` where

            - **preds**: :obj:`torch.FloatTensor` of model's prediction for the batch.
            - **targets**: :obj:`torch.Tensor` of model's targets (e.g. labels, target values).
        """
        input_texts, targets = batch
        _targets = targets
        targets = targets.to(textattack.shared.utils.device)

        if isinstance(model, transformers.PreTrainedModel):
            input_ids = tokenizer(
                input_texts,
                padding="max_length",
                return_tensors="pt",
                truncation=True,
            )
            input_ids.to(textattack.shared.utils.device)
            logits = model(**input_ids)[0]
        else:
            input_ids = tokenizer(input_texts)
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)
            input_ids = input_ids.to(textattack.shared.utils.device)
            logits = model(input_ids)

        if self.task_type == "regression":
            preds = logits
        else:
            preds = logits.argmax(dim=-1)

        return preds.cpu(), _targets

    def train(self):
        """Train the model on given training dataset."""
        if not self.train_dataset:
            raise ValueError("No `train_dataset` available for training.")

        textattack.shared.utils.set_seed(self.training_args.random_seed)
        if not os.path.exists(self.training_args.output_dir):
            os.makedirs(self.training_args.output_dir)

        # Save logger writes to file
        log_txt_path = os.path.join(self.training_args.output_dir, "train_log.txt")
        fh = logging.FileHandler(log_txt_path)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        logger.info(f"Writing logs to {log_txt_path}.")

        # Save original self.training_args to file
        args_save_path = os.path.join(
            self.training_args.output_dir, "training_args.json"
        )
        with open(args_save_path, "w", encoding="utf-8") as f:
            json.dump(self.training_args.__dict__, f)
        logger.info(f"Wrote original training args to {args_save_path}.")

        num_gpus = torch.cuda.device_count()
        tokenizer = self.model_wrapper.tokenizer
        model = self.model_wrapper.model

        if self.training_args.parallel and num_gpus > 1:
            # TODO: torch.nn.parallel.DistributedDataParallel
            # Supposedly faster than DataParallel, but requires more work to setup properly.
            model = torch.nn.DataParallel(model)
            logger.info(f"Training on {num_gpus} GPUs via `torch.nn.DataParallel`.")
            train_batch_size = self.training_args.per_device_train_batch_size * num_gpus
        else:
            train_batch_size = self.training_args.per_device_train_batch_size

        if self.attack is None:
            num_clean_epochs = self.training_args.num_epochs
        else:
            num_clean_epochs = self.training_args.num_clean_epochs

        total_clean_training_steps = (
            math.ceil(
                len(self.train_dataset)
                / (train_batch_size * self.training_args.gradient_accumulation_steps)
            )
            * num_clean_epochs
        )

        # calculate total_adv_training_data_length based on type of
        # num_train_adv_examples.
        # if num_train_adv_examples is float , num_train_adv_examples is a portion of train_dataset.
        if isinstance(self.training_args.num_train_adv_examples, float):
            total_adv_training_data_length = (
                len(self.train_dataset) * self.training_args.num_train_adv_examples
            )

        # if num_train_adv_examples is int and >=0 then it is taken as value.
        elif (
            isinstance(self.training_args.num_train_adv_examples, int)
            and self.training_args.num_train_adv_examples >= 0
        ):
            total_adv_training_data_length = self.training_args.num_train_adv_examples

        # if num_train_adv_examples is = -1 , we generate all possible adv examples.
        # Max number of all possible adv examples would be equal to train_dataset.
        else:
            total_adv_training_data_length = len(self.train_dataset)

        # Based on total_adv_training_data_length calculation , find total total_adv_training_steps
        total_adv_training_steps = math.ceil(
            (len(self.train_dataset) + total_adv_training_data_length)
            / (train_batch_size * self.training_args.gradient_accumulation_steps)
        ) * (self.training_args.num_epochs - num_clean_epochs)

        total_training_steps = total_clean_training_steps + total_adv_training_steps

        optimizer, scheduler = self.get_optimizer_and_scheduler(
            model, total_training_steps
        )

        self._print_training_args(
            total_training_steps, train_batch_size, num_clean_epochs
        )

        model.to(textattack.shared.utils.device)

        # Variables across epochs
        self._total_loss = 0.0
        self._current_loss = 0.0
        self._last_log_step = 0

        # `best_score` is used to keep track of the best model across training.
        # Could be loss, accuracy, or other metrics.
        best_eval_score = 0.0
        best_eval_score_epoch = 0
        best_model_path = None
        epochs_since_best_eval_score = 0

        for epoch in range(1, self.training_args.num_epochs + 1):
            logger.info("==========================================================")
            logger.info(f"Epoch {epoch}")

            if self.attack and epoch > num_clean_epochs:
                if (
                    epoch - num_clean_epochs - 1
                ) % self.training_args.attack_epoch_interval == 0:
                    # only generate a new adversarial training set every self.training_args.attack_period epochs after the clean epochs
                    # adv_dataset is instance of `textattack.datasets.Dataset`
                    model.eval()
                    adv_dataset = self._generate_adversarial_examples(epoch)
                    model.train()
                    model.to(textattack.shared.utils.device)
                else:
                    adv_dataset = None
            else:
                logger.info(f"Running clean epoch {epoch}/{num_clean_epochs}")
                adv_dataset = None

            train_dataloader = self.get_train_dataloader(
                self.train_dataset, adv_dataset, train_batch_size
            )
            model.train()
            # Epoch variables
            all_preds = []
            all_targets = []
            prog_bar = tqdm.tqdm(
                train_dataloader,
                desc="Iteration",
                position=0,
                leave=True,
                dynamic_ncols=True,
            )
            for step, batch in enumerate(prog_bar):
                loss, preds, targets = self.training_step(model, tokenizer, batch)

                if isinstance(model, torch.nn.DataParallel):
                    loss = loss.mean()

                loss = loss / self.training_args.gradient_accumulation_steps
                loss.backward()
                loss = loss.item()
                self._total_loss += loss
                self._current_loss += loss

                all_preds.append(preds)
                all_targets.append(targets)

                if (step + 1) % self.training_args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad()
                    self._global_step += 1

                if self._global_step > 0:
                    prog_bar.set_description(
                        f"Loss {self._total_loss/self._global_step:.5f}"
                    )

                # TODO: Better way to handle TB and Wandb logging
                if (self._global_step > 0) and (
                    self._global_step % self.training_args.logging_interval_step == 0
                ):
                    lr_to_log = (
                        scheduler.get_last_lr()[0]
                        if scheduler
                        else self.training_args.learning_rate
                    )
                    if self._global_step - self._last_log_step >= 1:
                        loss_to_log = round(
                            self._current_loss
                            / (self._global_step - self._last_log_step),
                            4,
                        )
                    else:
                        loss_to_log = round(self._current_loss, 4)

                    log = {"train/loss": loss_to_log, "train/learning_rate": lr_to_log}
                    if self.training_args.log_to_tb:
                        self._tb_log(log, self._global_step)

                    if self.training_args.log_to_wandb:
                        self._wandb_log(log, self._global_step)

                    self._current_loss = 0.0
                    self._last_log_step = self._global_step

                # Save model checkpoint to file.
                if self.training_args.checkpoint_interval_steps:
                    if (
                        self._global_step > 0
                        and (
                            self._global_step
                            % self.training_args.checkpoint_interval_steps
                        )
                        == 0
                    ):
                        self._save_model_checkpoint(
                            model, tokenizer, step=self._global_step
                        )

            preds = torch.cat(all_preds)
            targets = torch.cat(all_targets)
            if self._metric_name == "accuracy":
                correct_predictions = (preds == targets).sum().item()
                accuracy = correct_predictions / len(targets)
                metric_log = {"train/train_accuracy": accuracy}
                logger.info(f"Train accuracy: {accuracy*100:.2f}%")
            else:
                pearson_correlation, pearson_pvalue = scipy.stats.pearsonr(
                    preds, targets
                )
                metric_log = {
                    "train/pearson_correlation": pearson_correlation,
                    "train/pearson_pvalue": pearson_pvalue,
                }
                logger.info(f"Train Pearson correlation: {pearson_correlation:.4f}%")

            if len(targets) > 0:
                if self.training_args.log_to_tb:
                    self._tb_log(metric_log, epoch)
                if self.training_args.log_to_wandb:
                    metric_log["epoch"] = epoch
                    self._wandb_log(metric_log, self._global_step)

            # Evaluate after each epoch.
            eval_score = self.evaluate()

            if self.training_args.log_to_tb:
                self._tb_log({f"eval/{self._metric_name}": eval_score}, epoch)
            if self.training_args.log_to_wandb:
                self._wandb_log(
                    {f"eval/{self._metric_name}": eval_score, "epoch": epoch},
                    self._global_step,
                )

            if (
                self.training_args.checkpoint_interval_epochs
                and (epoch % self.training_args.checkpoint_interval_epochs) == 0
            ):
                self._save_model_checkpoint(model, tokenizer, epoch=epoch)

            if eval_score > best_eval_score:
                best_eval_score = eval_score
                best_eval_score_epoch = epoch
                epochs_since_best_eval_score = 0
                self._save_model_checkpoint(model, tokenizer, best=True)
                logger.info(
                    f"Best score found. Saved model to {self.training_args.output_dir}/best_model/"
                )
            else:
                epochs_since_best_eval_score += 1
                if self.training_args.early_stopping_epochs and (
                    epochs_since_best_eval_score
                    > self.training_args.early_stopping_epochs
                ):
                    logger.info(
                        f"Stopping early since it's been {self.training_args.early_stopping_epochs} steps since validation score increased."
                    )
                    break

        if self.training_args.log_to_tb:
            self._tb_writer.flush()

        # Finish training
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        if self.training_args.load_best_model_at_end:
            best_model_path = os.path.join(self.training_args.output_dir, "best_model")
            if hasattr(model, "from_pretrained"):
                model = model.__class__.from_pretrained(best_model_path)
            else:
                model = model.load_state_dict(
                    torch.load(os.path.join(best_model_path, "pytorch_model.bin"))
                )

        if self.training_args.save_last:
            self._save_model_checkpoint(model, tokenizer, last=True)

        self.model_wrapper.model = model
        self._write_readme(best_eval_score, best_eval_score_epoch, train_batch_size)

    def evaluate(self):
        """Evaluate the model on given evaluation dataset."""

        if not self.eval_dataset:
            raise ValueError("No `eval_dataset` available for training.")

        logging.info("Evaluating model on evaluation dataset.")
        model = self.model_wrapper.model
        tokenizer = self.model_wrapper.tokenizer

        model.eval()
        all_preds = []
        all_targets = []

        if isinstance(model, torch.nn.DataParallel):
            num_gpus = torch.cuda.device_count()
            eval_batch_size = self.training_args.per_device_eval_batch_size * num_gpus
        else:
            eval_batch_size = self.training_args.per_device_eval_batch_size

        eval_dataloader = self.get_eval_dataloader(self.eval_dataset, eval_batch_size)

        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                preds, targets = self.evaluate_step(model, tokenizer, batch)
                all_preds.append(preds)
                all_targets.append(targets)

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)

        if self.task_type == "regression":
            pearson_correlation, pearson_p_value = scipy.stats.pearsonr(preds, targets)
            eval_score = pearson_correlation
        else:
            correct_predictions = (preds == targets).sum().item()
            accuracy = correct_predictions / len(targets)
            eval_score = accuracy

        if self._metric_name == "accuracy":
            logger.info(f"Eval {self._metric_name}: {eval_score*100:.2f}%")
        else:
            logger.info(f"Eval {self._metric_name}: {eval_score:.4f}%")

        return eval_score

    def _write_readme(self, best_eval_score, best_eval_score_epoch, train_batch_size):
        if isinstance(self.training_args, CommandLineTrainingArgs):
            model_name = self.training_args.model_name_or_path
        elif isinstance(self.model_wrapper.model, transformers.PreTrainedModel):
            if (
                hasattr(self.model_wrapper.model.config, "_name_or_path")
                and self.model_wrapper.model.config._name_or_path in HUGGINGFACE_MODELS
            ):
                # TODO Better way than just checking HUGGINGFACE_MODELS ?
                model_name = self.model_wrapper.model.config._name_or_path
            elif hasattr(self.model_wrapper.model.config, "model_type"):
                model_name = self.model_wrapper.model.config.model_type
            else:
                model_name = ""
        else:
            model_name = ""

        if model_name:
            model_name = f"`{model_name}`"

        if (
            isinstance(self.training_args, CommandLineTrainingArgs)
            and self.training_args.model_max_length
        ):
            model_max_length = self.training_args.model_max_length
        elif isinstance(
            self.model_wrapper.model,
            (
                transformers.PreTrainedModel,
                LSTMForClassification,
                WordCNNForClassification,
            ),
        ):
            model_max_length = self.model_wrapper.tokenizer.model_max_length
        else:
            model_max_length = None

        if model_max_length:
            model_max_length_str = f" a maximum sequence length of {model_max_length},"
        else:
            model_max_length_str = ""

        if isinstance(
            self.train_dataset, textattack.datasets.HuggingFaceDataset
        ) and hasattr(self.train_dataset, "_name"):
            dataset_name = self.train_dataset._name
            if hasattr(self.train_dataset, "_subset"):
                dataset_name += f" ({self.train_dataset._subset})"
        elif isinstance(
            self.eval_dataset, textattack.datasets.HuggingFaceDataset
        ) and hasattr(self.eval_dataset, "_name"):
            dataset_name = self.eval_dataset._name
            if hasattr(self.eval_dataset, "_subset"):
                dataset_name += f" ({self.eval_dataset._subset})"
        else:
            dataset_name = None

        if dataset_name:
            dataset_str = (
                "and the `{dataset_name}` dataset loaded using the `datasets` library"
            )
        else:
            dataset_str = ""

        loss_func = (
            "mean squared error" if self.task_type == "regression" else "cross-entropy"
        )
        metric_name = (
            "pearson correlation" if self.task_type == "regression" else "accuracy"
        )
        epoch_info = f"{best_eval_score_epoch} epoch" + (
            "s" if best_eval_score_epoch > 1 else ""
        )
        readme_text = f"""
            ## TextAttack Model Card

            This {model_name} model was fine-tuned using TextAttack{dataset_str}. The model was fine-tuned
            for {self.training_args.num_epochs} epochs with a batch size of {train_batch_size},
            {model_max_length_str} and an initial learning rate of {self.training_args.learning_rate}.
            Since this was a {self.task_type} task, the model was trained with a {loss_func} loss function.
            The best score the model achieved on this task was {best_eval_score}, as measured by the
            eval set {metric_name}, found after {epoch_info}.

            For more information, check out [TextAttack on Github](https://github.com/QData/TextAttack).

            """

        readme_save_path = os.path.join(self.training_args.output_dir, "README.md")
        with open(readme_save_path, "w", encoding="utf-8") as f:
            f.write(readme_text.strip() + "\n")
        logger.info(f"Wrote README to {readme_save_path}.")
