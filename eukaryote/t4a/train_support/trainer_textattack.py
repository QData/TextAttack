import collections
import json
import logging
import math
import os

import scipy
import eukaryote
import torch
import tqdm
import transformers
from eukaryote.attack_args import AttackArgs
from eukaryote.attack_results import (
    MaximizedAttackResult,
    SuccessfulAttackResult,
)
from eukaryote.attacker import Attacker
from eukaryote.shared.utils import logger

# from eukaryote.trainer import Trainer # commented out bc linter claims it's unused, keeping in case it's actually needed


class T4A_Trainer(eukaryote.Trainer):
    """Fork of the built-in TextAttack `Trainer` class to optionally supress
    the model saving behavior, as well as add an optional callback after each
    training epoch.

    Note: all the methods are largely copied from the TextAttack source (i.e.
    forked).

    These functions can be further extended or replaced to implement, for
    example, early stopping with respect to a metric other than accuracy (which
    TextAttack has hard coded).
    """

    def __init__(
        self,
        *args,
        after_epoch_fn=None,
        disable_save=False,
        skip_eval=False,
        retain_adversarial_examples=False,
        num_train_examples=None,
        **kwargs,
    ):
        super(T4A_Trainer, self).__init__(*args, **kwargs)

        self.after_epoch_fn = after_epoch_fn
        self.disable_save = disable_save
        self.skip_eval = skip_eval
        self.retain_adversarial_examples = retain_adversarial_examples

        if num_train_examples is None:
            self.num_train_examples = len(self.train_dataset)
        elif isinstance(num_train_examples, float):
            self.num_train_examples = math.ceil(
                len(self.train_dataset) * num_train_examples
            )
        else:
            self.num_train_examples = num_train_examples

    def get_train_dataloader(self, dataset, adv_dataset, batch_size):
        """Returns the `torch.utils.data.DataLoader` for training.

        Note: forked for adding the following features:
            - correctly marking samples as adversarial (this is an upstream bug
              in TextAttack)
            - taking a sample of the training dataset
        """

        def collate_fn(data):
            input_texts = []
            targets = []
            is_adv_sample = []
            for item in data:
                _input, label = item

                if isinstance(_input, collections.OrderedDict):
                    if "adv" in _input:
                        is_adv_sample.append(True)
                        _input = tuple(v for k, v in _input.items() if k != "adv")
                    else:
                        is_adv_sample.append(False)
                        _input = tuple(_input.values())
                else:
                    _input = tuple(_input)
                    is_adv_sample.append(False)

                if len(_input) == 1:
                    _input = _input[0]
                input_texts.append(_input)
                targets.append(label)

            return (
                input_texts,
                torch.tensor(targets),
                torch.tensor(is_adv_sample),
            )

        if self.num_train_examples != len(dataset):
            # This method of randomly sampling the dataset could definitely be
            # made more efficient, but since the given `dataset` is already
            # loaded as `(OrderedDict, torch.Tensor)` pairs, this passes.
            sampler = torch.utils.data.RandomSampler(
                dataset, replacement=True, num_samples=self.num_train_examples
            )
            dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler)
            examples = [
                (tuple(v[0] for v in od.values()), label.item())
                for od, label in dataloader
            ]
            # Don't pass label_map as labels have already been mapped
            dataset = eukaryote.datasets.Dataset(
                examples,
                input_columns=dataset.input_columns,
                label_names=dataset.label_names,
                output_scale_factor=dataset.output_scale_factor,
                shuffle=False,
            )

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

    def training_step(self, model, tokenizer, batch):
        """Perform a single training step on a batch of inputs.

        Note: forked to also return logits and if sample is adversarial.
        """

        input_texts, targets, is_adv_sample = batch
        _targets = targets
        targets = targets.to(eukaryote.shared.utils.device)

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
            input_ids.to(eukaryote.shared.utils.device)
            logits = model(**input_ids)[0]
        else:
            input_ids = tokenizer(input_texts)
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)
            input_ids = input_ids.to(eukaryote.shared.utils.device)
            logits = model(input_ids)

        if self.task_type == "regression":
            loss = self.loss_fct(logits.squeeze(), targets.squeeze())
            preds = logits
        else:
            loss = self.loss_fct(logits, targets)
            preds = logits.argmax(dim=-1)

        sample_weights = torch.ones(
            is_adv_sample.size(), device=eukaryote.shared.utils.device
        )
        sample_weights[is_adv_sample] *= self.training_args.alpha
        loss = loss * sample_weights
        loss = torch.mean(loss)
        preds = preds.cpu()

        return loss, preds, logits, is_adv_sample, _targets

    def train(self):
        """Train the model on given training dataset.

        Note: forked for adding the following features:
            - disable automatic saving of various artifacts
            - optionally skip attack evaluation for speedup
            - call a callback after each epoch, including passing the training
              predictions, etc.
        """

        if not self.train_dataset:
            raise ValueError("No `train_dataset` available for training.")

        eukaryote.shared.utils.set_seed(self.training_args.random_seed)

        if not self.disable_save:
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
                self.num_train_examples
                / (train_batch_size * self.training_args.gradient_accumulation_steps)
            )
            * num_clean_epochs
        )
        total_adv_training_steps = math.ceil(
            (self.num_train_examples + self.training_args.num_train_adv_examples)
            / (train_batch_size * self.training_args.gradient_accumulation_steps)
        ) * (self.training_args.num_epochs - num_clean_epochs)

        total_training_steps = total_clean_training_steps + total_adv_training_steps

        optimizer, scheduler = self.get_optimizer_and_scheduler(
            model, total_training_steps
        )

        self._print_training_args(
            total_training_steps, train_batch_size, num_clean_epochs
        )

        model.to(eukaryote.shared.utils.device)

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

        adv_dataset = None

        for epoch in range(1, self.training_args.num_epochs + 1):
            logger.info("==========================================================")
            logger.info(f"Epoch {epoch}")

            if self.attack and epoch > num_clean_epochs:
                if (
                    epoch - num_clean_epochs - 1
                ) % self.training_args.attack_epoch_interval == 0:
                    # only generate a new adversarial training set every self.training_args.attack_period epochs after the clean epochs
                    # adv_dataset is instance of `eukaryote.datasets.Dataset`
                    model.eval()
                    adv_dataset = self._generate_adversarial_examples(epoch)
                    model.train()
                    model.to(eukaryote.shared.utils.device)
                elif not self.retain_adversarial_examples:
                    adv_dataset = None
            else:
                logger.info(f"Running clean epoch {epoch}/{num_clean_epochs}")
                if not self.retain_adversarial_examples:
                    adv_dataset = None

            train_dataloader = self.get_train_dataloader(
                self.train_dataset, adv_dataset, train_batch_size
            )
            model.train()
            # Epoch variables
            all_preds = []
            all_logits = []
            all_is_adv_sample = []
            all_targets = []
            prog_bar = tqdm.tqdm(
                train_dataloader,
                desc="Iteration",
                position=0,
                leave=True,
                dynamic_ncols=True,
            )
            for step, batch in enumerate(prog_bar):
                (
                    loss,
                    preds,
                    logits,
                    is_adv_sample,
                    targets,
                ) = self.training_step(model, tokenizer, batch)

                if isinstance(model, torch.nn.DataParallel):
                    loss = loss.mean()

                loss = loss / self.training_args.gradient_accumulation_steps
                loss.backward()
                loss = loss.item()
                self._total_loss += loss
                self._current_loss += loss

                all_preds.append(preds)
                all_logits.append(logits)
                all_is_adv_sample.append(is_adv_sample)
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

                    log = {
                        "train/loss": loss_to_log,
                        "train/learning_rate": lr_to_log,
                    }
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
                        and not self.disable_save
                    ):
                        self._save_model_checkpoint(
                            model, tokenizer, step=self._global_step
                        )

            preds = torch.cat(all_preds)
            logits = torch.cat(all_logits)
            is_adv_sample = torch.cat(all_is_adv_sample)
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

            if not self.skip_eval:
                # Evaluate after each epoch.
                eval_score = self.evaluate()

                if self.training_args.log_to_tb:
                    self._tb_log({f"eval/{self._metric_name}": eval_score}, epoch)
                if self.training_args.log_to_wandb:
                    self._wandb_log(
                        {
                            f"eval/{self._metric_name}": eval_score,
                            "epoch": epoch,
                        },
                        self._global_step,
                    )

                if (
                    self.training_args.checkpoint_interval_epochs
                    and (epoch % self.training_args.checkpoint_interval_epochs) == 0
                    and not self.disable_save
                ):
                    self._save_model_checkpoint(model, tokenizer, epoch=epoch)

                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    best_eval_score_epoch = epoch
                    epochs_since_best_eval_score = 0
                    if not self.disable_save:
                        self._save_model_checkpoint(model, tokenizer, best=True)
                        logger.info(
                            f"Best score found. Saved model to {self.training_args.output_dir}/best_model/"
                        )
                    else:
                        logger.info("Best score found")
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

            if self.after_epoch_fn:
                self.model_wrapper.model = model
                self.after_epoch_fn(
                    epoch,
                    {
                        "preds": preds,
                        "logits": logits,
                        "is_adv_sample": is_adv_sample,
                        "targets": targets,
                    },
                )

        if self.training_args.log_to_tb:
            self._tb_writer.flush()

        # Finish training
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        if not self.disable_save:
            if self.training_args.load_best_model_at_end:
                best_model_path = os.path.join(
                    self.training_args.output_dir, "best_model"
                )
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

    def _generate_adversarial_examples(self, epoch):
        """Generate adversarial examples using attacker.

        Note: forked to correctly mark samples as adversarial (this is an
        upstream bug in TextAttack).
        """

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

        attack_args = AttackArgs(
            num_successful_examples=num_train_adv_examples,
            num_examples_offset=0,
            query_budget=self.training_args.query_budget_train,
            shuffle=True,
            parallel=self.training_args.parallel,
            num_workers_per_device=self.training_args.attack_num_workers_per_device,
            disable_stdout=True,
            silent=True,
            log_to_txt=log_file_name + ".txt" if not self.disable_save else None,
            log_to_csv=log_file_name + ".csv" if not self.disable_save else None,
        )

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
        adversarial_examples = [
            (
                (
                    tuple(r.perturbed_result.attacked_text._text_input.values())
                    + (True,)
                ),
                r.perturbed_result.ground_truth_output,
            )
            for r in results
            if isinstance(r, (SuccessfulAttackResult, MaximizedAttackResult))
        ]
        adversarial_dataset = eukaryote.datasets.Dataset(
            adversarial_examples,
            input_columns=tuple(self.train_dataset.input_columns) + ("adv",),
            label_map=self.train_dataset.label_map,
            label_names=self.train_dataset.label_names,
            output_scale_factor=self.train_dataset.output_scale_factor,
            shuffle=False,
        )
        return adversarial_dataset
