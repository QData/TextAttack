import collections
import functools
import json
import logging
import math
import os

import numpy as np
import scipy
import torch
import tqdm
import transformers

import textattack
from textattack.models.helpers import LSTMForClassification, WordCNNForClassification
from textattack.models.wrappers import ModelWrapper

from .attack import Attack
from .attack_args import AttackArgs
from .attacker import Attacker
from .model_args import HUGGINGFACE_MODELS
from .training_args import CommandLineTrainingArgs, TrainingArgs

logger = textattack.shared.logger


# Helper functions for collating data
def collate_fn(input_columns, data):
    input_texts = []
    labels = []
    for _input, label in data:
        _input = tuple(_input[c] for c in input_columns)
        if len(_input) == 1:
            _input = _input[0]
        input_texts.append(_input)
        labels.append(label)
    return input_texts, torch.tensor(labels)


class Trainer:
    """Trainer is training and eval loop for adversarial training.

    It is designed to work with PyTorch and Transformers models.
    """

    def __init__(
        self,
        model_wrapper,
        task_type,
        attack,
        train_dataset,
        eval_dataset,
        training_args,
    ):
        assert isinstance(
            model_wrapper, ModelWrapper
        ), f"`model_wrapper` must be of type `textattack.models.wrappers.ModelWrapper`, but got type `{type(model_wrapper)}`."
        # TODO: Support seq2seq training
        assert task_type in {
            "classification",
            "regression",
        }, '`task_type` must either be "classification" or "regression"'
        assert isinstance(
            attack, Attack
        ), f"`attack` argument must be of type `textattack.Attack`, but got type of `{type(attack)}`."
        assert isinstance(
            train_dataset, textattack.datasets.Dataset
        ), f"`train_dataset` must be of type `textattack.datasets.Dataset`, but got type `{type(train_dataset)}`."
        assert isinstance(
            eval_dataset, textattack.datasets.Dataset
        ), f"`eval_dataset` must be of type `textattack.datasets.Dataset`, but got type `{type(eval_dataset)}`."
        assert isinstance(
            training_args, TrainingArgs
        ), f"`training_args` must be of type `textattack.TrainingArgs`, but got type `{type(training_args)}`."

        if id(model_wrapper) != id(attack.goal_function.model):
            logger.warn(
                "WARNING: `model_wrapper` and the victim model of `attack` is not the same model."
            )

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

    def _generate_adversarial_examples(self, dataset, epoch, eval_mode=False):
        """Generate adversarial examples using attacker."""
        if eval_mode:
            logger.info("Attacking model to evaluate adversarial robustness...")
        else:
            logger.info("Attacking model to generate new adversarial training set...")

        num_examples = (
            self.training_args.num_eval_adv_examples
            if eval_mode
            else self.training_args.num_train_adv_examples
        )
        query_budget = (
            self.training_args.query_budget_eval
            if eval_mode
            else self.training_args.query_budget_train
        )
        shuffle = False if eval_mode else True
        base_file_name = (
            f"attack-eval-{epoch}" if eval_mode else f"attack-train-{epoch}"
        )
        log_file_name = os.path.join(self.training_args.output_dir, base_file_name)
        attack_args = AttackArgs(
            num_examples=num_examples,
            num_examples_offset=0,
            query_budget=query_budget,
            shuffle=shuffle,
            attack_n=True,
            parallel=self.training_args.parallel,
            num_workers_per_device=self.training_args.attack_num_workers_per_device,
            disable_stdout=True,
            silent=True,
            ignore_exceptions=True,
            log_to_txt=log_file_name + ".txt",
            log_to_csv=log_file_name + ".csv",
        )
        attacker = Attacker(self.attack, dataset, attack_args)
        results = attacker.attack_dataset()

        if eval_mode:
            return results
        else:
            attack_types = collections.Counter(r.__class__.__name__ for r in results)
            total_attacks = (
                attack_types["SuccessfulAttackResult"]
                + attack_types["FailedAttackResult"]
            )
            success_rate = attack_types["SuccessfulAttackResult"] / total_attacks * 100
            logger.info(
                f"Attack Success Rate: {success_rate:.2f}% [{attack_types['SuccessfulAttackResult']} / {total_attacks}]"
            )
            adversarial_examples = [
                (
                    tuple(r.perturbed_result.attacked_text._text_input.values()),
                    r.perturbed_result.ground_truth_output,
                )
                for r in results
            ]
            adversarial_dataset = textattack.datasets.Dataset(
                adversarial_examples,
                input_columns=dataset.input_columns,
                label_map=dataset.label_map,
                label_names=dataset.label_names,
                output_scale_factor=dataset.output_scale_factor,
                shuffle=False,
            )
            return adversarial_dataset

    def _training_setup(self):
        """Handle all the training set ups including logging."""
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

    def _print_training_args(self, total_training_steps):
        logger.info("==================== Running Training ====================")
        logger.info(f"Num epochs = {self.training_args.num_epochs}")
        logger.info(f"Num clean epochs = {self.training_args.num_clean_epochs}")
        logger.info(f"Num total steps = {total_training_steps}")
        logger.info(f"Num training examples = {len(self.train_dataset)}")
        logger.info(f"Num evaluation examples = {len(self.eval_dataset)}")
        logger.info(f"Starting learning rate = {self.training_args.learning_rate}")
        logger.info(f"Num warmup steps = {self.training_args.num_warmup_steps}")
        logger.info(f"Weight decay = {self.training_args.weight_decay}")

    def _get_tensorboard_writer(self):
        from torch.utils.tensorboard import SummaryWriter

        tb_writer = SummaryWriter(self.training_args.tb_log_dir)
        tb_writer.add_hparams(self.training_args.__dict__, {})
        tb_writer.flush()
        return tb_writer

    def _init_wandb(self):
        global wandb
        import wandb

        wandb.init(
            project=self.training_args.wand_project, config=self.training_args.__dict__
        )

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

    def _get_optimizer_and_scheduler(self, model, total_training_steps):
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

            scheduler = transformers.optimization.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.training_args.num_warmup_steps,
                num_training_steps=total_training_steps,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda x: x.requires_grad, model.parameters()),
                lr=self.training_args.learning_rate,
            )
            scheduler = None

        return optimizer, scheduler

    def train(self):
        self._training_setup()
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

        model.to(textattack.shared.utils.device)
        total_training_steps = (
            math.ceil(
                len(self.train_dataset)
                / (train_batch_size * self.training_args.gradient_accumulation_steps)
            )
            * self.training_args.num_epochs
        )

        if self.training_args.log_to_tb:
            tb_writer = self._get_tensorboard_writer()
        if self.training_args.log_to_wandb:
            self._init_wandb()

        optimizer, scheduler = self._get_optimizer_and_scheduler(
            model, total_training_steps
        )

        collate_func = functools.partial(collate_fn, self.train_dataset.input_columns)

        if self.task_type == "regression":
            loss_fct = torch.nn.MSELoss()
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        # Variables across epochs
        global_step = 0
        total_loss = 0.0
        # `best_score` is used to keep track of the best model across training.
        # Could be loss, accuracy, or other metrics.
        best_eval_score = 0.0
        best_eval_score_epoch = 0
        best_model_path = None
        epochs_since_best_eval_score = 0
        self._print_training_args(total_training_steps)
        for epoch in range(1, self.training_args.num_epochs + 1):
            logger.info("==========================================================")
            logger.info(f"Epoch {epoch}")
            # Adversarial attack and DataLoader creation
            if epoch > self.training_args.num_clean_epochs:
                if (
                    epoch - self.training_args.num_clean_epochs - 1
                ) % self.training_args.attack_epoch_interval == 0:
                    # only generate a new adversarial training set every self.training_args.attack_period epochs
                    # after the clean epochs
                    # adv_example_dataset is instance of `textattack.datasets.Dataset
                    model.eval()
                    model.cpu()
                    adv_example_dataset = self._generate_adversarial_examples(
                        self.train_dataset, epoch
                    )
                    train_dataset = torch.utils.data.ConcatDataset(
                        [self.train_dataset, adv_example_dataset]
                    )
                    model.to(textattack.shared.utils.device)
                    model.train()
                else:
                    train_dataset = self.train_dataset
            else:
                logger.info(
                    f"Running clean epoch {epoch}/{self.training_args.num_clean_epochs}"
                )
                train_dataset = self.train_dataset

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                shuffle=True,
                collate_fn=collate_func,
            )
            model.train()
            # Epoch-specific variables
            correct_predictions = 0
            total_predictions = 0
            prog_bar = tqdm.tqdm(
                train_dataloader, desc="Iteration", position=0, leave=True
            )
            for step, batch in enumerate(prog_bar):
                input_texts, labels = batch
                labels = labels.to(textattack.shared.utils.device)
                if isinstance(model, transformers.PreTrainedModel):
                    input_ids = tokenizer(
                        input_texts,
                        padding="max_length",
                        return_tensors="pt",
                        truncation=True,
                    )
                    for key in input_ids:
                        if isinstance(input_ids[key], torch.Tensor):
                            input_ids[key] = input_ids[key].to(
                                textattack.shared.utils.device
                            )
                    logits = model(**input_ids)[0]
                else:
                    input_ids = tokenizer(input_texts)
                    if not isinstance(input_ids, torch.Tensor):
                        input_ids = torch.tensor(input_ids)
                    input_ids = input_ids.to(textattack.shared.utils.device)
                    logits = model(input_ids)

                if self.task_type == "regression":
                    # TODO integrate with textattack `metrics` package
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
                    pred_labels = logits.argmax(dim=-1)
                    correct_predictions += (pred_labels == labels).sum().item()
                    total_predictions += len(pred_labels)

                if isinstance(model, torch.nn.DataParallel):
                    loss = loss.mean()
                if self.training_args.gradient_accumulation_steps > 1:
                    loss = loss / self.training_args.gradient_accumulation_steps
                loss.backward()

                total_loss += loss.item()

                if (step + 1) % self.training_args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                # TODO: Better way to handle TB and Wandb logging
                if global_step % self.training_args.logging_interval_step == 0:
                    lr_to_log = (
                        scheduler.get_last_lr()[0]
                        if scheduler
                        else self.training_args.learning_rate
                    )
                    if self.training_args.log_to_tb:
                        tb_writer.add_scalar("loss", loss.item(), global_step)
                        tb_writer.add_scalar("lr", lr_to_log, global_step)

                    if self.training_args.log_to_wandb:
                        wandb.log({"loss": loss.item()}, step=global_step)
                        wandb.log({"lr": lr_to_log}, step=global_step)

                if global_step > 0:
                    prog_bar.set_description(f"Loss {total_loss/global_step:.5f}")

                # Save model checkpoint to file.
                if self.training_args.checkpoint_interval_steps:
                    if (
                        global_step > 0
                        and self.training_args.checkpoint_interval_steps > 0
                        and (global_step % self.training_args.checkpoint_interval_steps)
                        == 0
                    ):
                        self._save_model_checkpoint(model, tokenizer, step=global_step)

            # Print training accuracy, if we're tracking it.
            if total_predictions > 0:
                train_acc = correct_predictions / total_predictions
                logger.info(f"Train accuracy: {train_acc*100:.2f}%")
                if self.training_args.log_to_tb:
                    tb_writer.add_scalar("epoch_train_acc", train_acc, global_step)
                if self.training_args.log_to_wandb:
                    wandb.log({"epoch_train_acc": train_acc}, step=global_step)

            # Check eval accuracy after each epoch.
            eval_score = self._evaluate(model, tokenizer)
            logger.info(
                f"Eval {'pearson correlation' if self.task_type == 'regression' else 'accuracy'}: {eval_score*100:.2f}%"
            )
            if self.training_args.log_to_tb:
                tb_writer.add_scalar("epoch_eval_score", eval_score, global_step)
            if self.training_args.log_to_wandb:
                wandb.log({"epoch_eval_score": eval_score}, step=global_step)

            if (
                self.training_args.checkpoint_interval_epochs
                and ((epoch - 1) % self.training_args.checkpoint_interval_epochs) == 0
            ):
                self._save_model_checkpoint(model, tokenizer, epoch=epoch)

            if eval_score > best_eval_score:
                best_eval_score = eval_score
                best_eval_score_epoch = epoch
                epochs_since_best_eval_score = 0
                self._save_model_checkpoint(model, tokenizer, best=True)
                logger.info(
                    f"Best acc found. Saved model to {self.training_args.output_dir}/best_model/"
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

            if self.training_args.eval_adversarial_robustness and (
                epoch >= self.training_args.num_clean_epochs
            ):
                # Evaluate adversarial robustness
                model.eval()
                model.cpu()
                adv_attack_results = self._generate_adversarial_examples(
                    self.eval_dataset, epoch, eval_mode=True
                )
                model.to(textattack.shared.utils.device)
                model.train()
                attack_types = [r.__class__.__name__ for r in adv_attack_results]
                attack_types = collections.Counter(attack_types)
                total_attacks = (
                    attack_types["SuccessfulAttackResult"]
                    + attack_types["FailedAttackResult"]
                )
                adv_succ_rate = attack_types["SuccessfulAttackResult"] / total_attacks
                num_queries = np.array(
                    [
                        r.num_queries
                        for r in adv_attack_results
                        if not isinstance(
                            r, textattack.attack_results.SkippedAttackResult
                        )
                    ]
                )
                avg_num_queries = round(num_queries.mean(), 2)

                if self.training_args.log_to_tb:
                    tb_writer.add_scalar(
                        "robustness_total_attacks", total_attacks, global_step
                    )
                    tb_writer.add_scalar(
                        "robustness_attack_succ_rate", adv_succ_rate, global_step
                    )
                    tb_writer.add_scalar(
                        "robustness_avg_num_queries", avg_num_queries, global_step
                    )
                if self.training_args.log_to_wandb:
                    wandb.log(
                        {"robustness_total_attacks": total_attacks}, step=global_step
                    )
                    wandb.log(
                        {"robustness_attack_succ_rate": adv_succ_rate}, step=global_step
                    )
                    wandb.log(
                        {"robustness_avg_num_queries": avg_num_queries},
                        step=global_step,
                    )

                logger.info(f"Eval total attack: {total_attacks}")
                logger.info(f"Eval attack success rate: {100*adv_succ_rate:.2f}%")
                logger.info(f"Eval avg num queries: {avg_num_queries}")

            if self.training_args.log_to_tb:
                tb_writer.flush()

        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        if self.training_args.load_best_model_at_end:
            best_model_path = os.path.join(self.training_args.output_dir, "best_model")
            if hasattr(model, "from_pretrained"):
                model = model.__class__.from_pretrained(best_model_path)
            else:
                model = model.load_state_dict(
                    torch.load(os.path.join(best_model_path, "model.pt"))
                )

        if self.training_args.save_last:
            self._save_model_checkpoint(model, tokenizer, last=True)

        self.model_wrapper.model = model
        self.write_readme(best_eval_score, best_eval_score_epoch, train_batch_size)

    def _evaluate(self, model, tokenizer):
        model.eval()
        correct = 0
        logits = []
        labels = []

        if isinstance(model, torch.nn.DataParallel):
            num_gpus = torch.cuda.device_count()
            eval_batch_size = self.training_args.per_device_eval_batch_size * num_gpus
        else:
            eval_batch_size = self.training_args.per_device_eval_batch_size

        collate_func = functools.partial(collate_fn, self.eval_dataset.input_columns)
        eval_dataloader = torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=eval_batch_size,
            collate_fn=collate_func,
        )

        with torch.no_grad():
            for input_texts, batch_labels in eval_dataloader:
                batch_labels = batch_labels.to(textattack.shared.utils.device)
                if isinstance(model, transformers.PreTrainedModel):
                    input_ids = tokenizer(
                        input_texts,
                        padding="max_length",
                        return_tensors="pt",
                        truncation=True,
                    )
                    for key in input_ids:
                        if isinstance(input_ids[key], torch.Tensor):
                            input_ids[key] = input_ids[key].to(
                                textattack.shared.utils.device
                            )
                    batch_logits = model(**input_ids)[0]
                else:
                    input_ids = tokenizer(input_texts)
                    if not isinstance(input_ids, torch.Tensor):
                        input_ids = torch.tensor(input_ids)
                    input_ids = input_ids.to(textattack.shared.utils.device)
                    batch_logits = model(input_ids)

                logits.extend(batch_logits.cpu().squeeze().tolist())
                labels.extend(batch_labels)

        model.train()
        logits = torch.tensor(logits)
        labels = torch.tensor(labels)

        if self.task_type == "regression":
            pearson_correlation, pearson_p_value = scipy.stats.pearsonr(logits, labels)
            return pearson_correlation
        else:
            preds = logits.argmax(dim=1)
            correct = (preds == labels).sum()
            return float(correct) / len(labels)

    def evaluate(self):
        logging.info("Evaluating model on evaluation dataset.")
        model = self.model_wrapper.model
        tokenizer = self.model_wrapper.tokenizer
        return self._evaluate(model, tokenizer)

    def write_readme(self, best_eval_score, best_eval_score_epoch, train_batch_size):
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
        elif isinstance(self.model_wrapper.model, transformers.PreTrainedModel):
            model_max_length = self.model_wrapper.config.max_position_embeddings
        elif isinstance(
            self.model_wrapper.model, (LSTMForClassification, WordCNNForClassification)
        ):
            model_max_length = self.model_wrapper.model.max_length
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
