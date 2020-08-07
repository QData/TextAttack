import collections
import json
import logging
import math
import os
import random

import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader, RandomSampler
import tqdm
import transformers

import textattack

from .train_args_helpers import (
    attack_from_args,
    augmenter_from_args,
    dataset_from_args,
    model_from_args,
    write_readme,
)

device = textattack.shared.utils.device
logger = textattack.shared.logger


def _save_args(args, save_path):
    """Dump args dictionary to a json.

    :param: args. Dictionary of arguments to save.
    :save_path: Path to json file to write args to.
    """
    final_args_dict = {k: v for k, v in vars(args).items() if _is_writable_type(v)}
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(final_args_dict, indent=2) + "\n")


def _get_sample_count(*lsts):
    """Get sample count of a dataset.

    :param *lsts: variable number of lists.
    :return: sample count of this dataset, if all lists match, else None.
    """
    if all(len(lst) == len(lsts[0]) for lst in lsts):
        sample_count = len(lsts[0])
    else:
        sample_count = None
    return sample_count


def _random_shuffle(*lsts):
    """Randomly shuffle a dataset. Applies the same permutation to each list
    (to preserve mapping between inputs and targets).

    :param *lsts: variable number of lists to shuffle.
    :return: shuffled lsts.
    """
    permutation = np.random.permutation(len(lsts[0]))
    shuffled = []
    for lst in lsts:
        shuffled.append((np.array(lst)[permutation]).tolist())
    return tuple(shuffled)


def _train_val_split(*lsts, split_val=0.2):
    """Split dataset into training and validation sets.

    :param *lsts: variable number of lists that make up a dataset (e.g. text, labels)
    :param split_val: float [0., 1.). Fraction of the dataset to reserve for evaluation.
    :return: (train split of list for list in lsts), (val split of list for list in lsts)
    """
    sample_count = _get_sample_count(*lsts)
    if not sample_count:
        raise Exception(
            "Batch Axis inconsistent. All input arrays must have first axis of equal length."
        )
    lsts = _random_shuffle(*lsts)
    split_idx = math.floor(sample_count * split_val)
    train_set = [lst[split_idx:] for lst in lsts]
    val_set = [lst[:split_idx] for lst in lsts]
    if len(train_set) == 1 and len(val_set) == 1:
        train_set = train_set[0]
        val_set = val_set[0]
    return train_set, val_set


def _filter_labels(text, labels, allowed_labels):
    """Keep examples with approved labels.

    :param text: list of text inputs.
    :param labels: list of corresponding labels.
    :param allowed_labels: list of approved label values.

    :return: (final_text, final_labels). Filtered version of text and labels
    """
    final_text, final_labels = [], []
    for text, label in zip(text, labels):
        if label in allowed_labels:
            final_text.append(text)
            final_labels.append(label)
    return final_text, final_labels


def _save_model_checkpoint(model, output_dir, global_step):
    """Save model checkpoint to disk.

    :param model: Model to save (pytorch)
    :param output_dir: Path to model save dir.
    :param global_step: Current global training step #. Used in ckpt filename.
    """
    # Save model checkpoint
    output_dir = os.path.join(output_dir, "checkpoint-{}".format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)


def _save_model(model, output_dir, weights_name, config_name):
    """Save model to disk.

    :param model: Model to save (pytorch)
    :param output_dir: Path to model save dir.
    :param weights_name: filename for model parameters.
    :param config_name: filename for config.
    """
    model_to_save = model.module if hasattr(model, "module") else model

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(output_dir, weights_name)
    output_config_file = os.path.join(output_dir, config_name)

    torch.save(model_to_save.state_dict(), output_model_file)
    try:
        model_to_save.config.to_json_file(output_config_file)
    except AttributeError:
        # no config
        pass


def _get_eval_score(model, eval_dataloader, do_regression):
    """Measure performance of a model on the evaluation set.

    :param model: Model to test.
    :param eval_dataloader: a torch DataLoader that iterates through the eval set.
    :param do_regression: bool. Whether we are doing regression (True) or classification (False)

    :return: pearson correlation, if do_regression==True, else classification accuracy [0., 1.]
    """
    model.eval()
    correct = 0
    logits = []
    labels = []
    for input_ids, batch_labels in eval_dataloader:
        batch_labels = batch_labels.to(device)
        if isinstance(input_ids, dict):
            ## dataloader collates dict backwards. This is a workaround to get
            # ids in the right shape for HuggingFace models
            input_ids = {k: torch.stack(v).T.to(device) for k, v in input_ids.items()}
            with torch.no_grad():
                batch_logits = model(**input_ids)[0]
        else:
            input_ids = input_ids.to(device)
            with torch.no_grad():
                batch_logits = model(input_ids)

        logits.extend(batch_logits.cpu().squeeze().tolist())
        labels.extend(batch_labels)

    model.train()
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    if do_regression:
        pearson_correlation, pearson_p_value = scipy.stats.pearsonr(logits, labels)
        return pearson_correlation
    else:
        preds = logits.argmax(dim=1)
        correct = (preds == labels).sum()
        return float(correct) / len(labels)


def _make_directories(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def _is_writable_type(obj):
    for ok_type in [bool, int, str, float]:
        if isinstance(obj, ok_type):
            return True
    return False


def batch_encode(tokenizer, text_list):
    if hasattr(tokenizer, "batch_encode"):
        return tokenizer.batch_encode(text_list)
    else:
        return [tokenizer.encode(text_input) for text_input in text_list]


def _make_dataloader(tokenizer, text, labels, batch_size):
    """Create torch DataLoader from list of input text and labels.

    :param tokenizer: Tokenizer to use for this text.
    :param text: list of input text.
    :param labels: list of corresponding labels.
    :param batch_size: batch size (int).
    :return: torch DataLoader for this training set.
    """
    text_ids = batch_encode(tokenizer, text)
    input_ids = np.array(text_ids)
    labels = np.array(labels)
    data = list((ids, label) for ids, label in zip(input_ids, labels))
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader


def _data_augmentation(text, labels, augmenter):
    """Use an augmentation method to expand a training set.

    :param text: list of input text.
    :param labels: list of corresponding labels.
    :param augmenter: textattack.augmentation.Augmenter, augmentation scheme.

    :return: augmented_text, augmented_labels. list of (augmented) input text and labels.
    """
    aug_text = augmenter.augment_many(text)
    # flatten augmented examples and duplicate labels
    flat_aug_text = []
    flat_aug_labels = []
    for i, examples in enumerate(aug_text):
        for aug_ver in examples:
            flat_aug_text.append(aug_ver)
            flat_aug_labels.append(labels[i])
    return flat_aug_text, flat_aug_labels


def _generate_adversarial_examples(model, attack_class, dataset):
    """Create a dataset of adversarial examples based on perturbations of the
    existing dataset.

    :param model: Model to attack.
    :param attack_class: class name of attack recipe to run.
    :param dataset: iterable of (text, label) pairs.

    :return: list(AttackResult) of adversarial examples.
    """
    attack = attack_class(model)

    # Fix TensorFlow GPU memory growth
    import tensorflow as tf

    tf.get_logger().setLevel("WARNING")
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    adv_attack_results = []
    for adv_ex in tqdm.tqdm(
        attack.attack_dataset(dataset), desc="Attack", total=len(dataset)
    ):
        adv_attack_results.append(adv_ex)
    return adv_attack_results


def train_model(args):
    textattack.shared.utils.set_seed(args.random_seed)

    logger.warn(
        "WARNING: TextAttack's model training feature is in beta. Please report any issues on our Github page, https://github.com/QData/TextAttack/issues."
    )
    _make_directories(args.output_dir)

    num_gpus = torch.cuda.device_count()

    # Save logger writes to file
    log_txt_path = os.path.join(args.output_dir, "log.txt")
    fh = logging.FileHandler(log_txt_path)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.info(f"Writing logs to {log_txt_path}.")

    # Get list of text and list of label (integers) from disk.
    train_text, train_labels, eval_text, eval_labels = dataset_from_args(args)

    # Filter labels
    if args.allowed_labels:
        train_text, train_labels = _filter_labels(
            train_text, train_labels, args.allowed_labels
        )
        eval_text, eval_labels = _filter_labels(
            eval_text, eval_labels, args.allowed_labels
        )

    if args.pct_dataset < 1.0:
        logger.info(f"Using {args.pct_dataset*100}% of the training set")
        (train_text, train_labels), _ = _train_val_split(
            train_text, train_labels, split_val=1.0 - args.pct_dataset
        )
    train_examples_len = len(train_text)

    # data augmentation
    augmenter = augmenter_from_args(args)
    if augmenter:
        logger.info(f"Augmenting {len(train_text)} samples with {augmenter}")
        train_text, train_labels = _data_augmentation(
            train_text, train_labels, augmenter
        )

    # label_id_len = len(train_labels)
    label_set = set(train_labels)
    args.num_labels = len(label_set)
    logger.info(f"Loaded dataset. Found: {args.num_labels} labels: {sorted(label_set)}")

    if isinstance(train_labels[0], float):
        # TODO come up with a more sophisticated scheme for knowing when to do regression
        logger.warn("Detected float labels. Doing regression.")
        args.num_labels = 1
        args.do_regression = True
    else:
        args.do_regression = False

    if len(train_labels) != len(train_text):
        raise ValueError(
            f"Number of train examples ({len(train_text)}) does not match number of labels ({len(train_labels)})"
        )
    if len(eval_labels) != len(eval_text):
        raise ValueError(
            f"Number of teste xamples ({len(eval_text)}) does not match number of labels ({len(eval_labels)})"
        )

    model_wrapper = model_from_args(args, args.num_labels)
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer

    attack_class = attack_from_args(args)
    # We are adversarial training if the user specified an attack along with
    # the training args.
    adversarial_training = (attack_class is not None) and (not args.check_robustness)

    # multi-gpu training
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
        logger.info("Using torch.nn.DataParallel.")
    logger.info(f"Training model across {num_gpus} GPUs")

    num_train_optimization_steps = (
        int(train_examples_len / args.batch_size / args.grad_accum_steps)
        * args.num_train_epochs
    )

    if args.model == "lstm" or args.model == "cnn":

        def need_grad(x):
            return x.requires_grad

        optimizer = torch.optim.Adam(
            filter(need_grad, model.parameters()), lr=args.learning_rate
        )
        scheduler = None
    else:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = transformers.optimization.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate
        )

        scheduler = transformers.optimization.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_proportion,
            num_training_steps=num_train_optimization_steps,
        )

    # Start Tensorboard and log hyperparams.
    from torch.utils.tensorboard import SummaryWriter

    tb_writer = SummaryWriter(args.output_dir)

    # Use Weights & Biases, if enabled.
    if args.enable_wandb:
        global wandb
        import wandb

        wandb.init(sync_tensorboard=True)

    # Save original args to file
    args_save_path = os.path.join(args.output_dir, "train_args.json")
    _save_args(args, args_save_path)
    logger.info(f"Wrote original training args to {args_save_path}.")

    tb_writer.add_hparams(
        {k: v for k, v in vars(args).items() if _is_writable_type(v)}, {}
    )

    # Start training
    logger.info("***** Running training *****")
    if augmenter:
        logger.info(f"\tNum original examples = {train_examples_len}")
        logger.info(f"\tNum examples after augmentation = {len(train_text)}")
    else:
        logger.info(f"\tNum examples = {train_examples_len}")
    logger.info(f"\tBatch size = {args.batch_size}")
    logger.info(f"\tMax sequence length = {args.max_length}")
    logger.info(f"\tNum steps = {num_train_optimization_steps}")
    logger.info(f"\tNum epochs = {args.num_train_epochs}")
    logger.info(f"\tLearning rate = {args.learning_rate}")

    eval_dataloader = _make_dataloader(
        tokenizer, eval_text, eval_labels, args.batch_size
    )
    train_dataloader = _make_dataloader(
        tokenizer, train_text, train_labels, args.batch_size
    )

    global_step = 0
    tr_loss = 0

    model.train()
    args.best_eval_score = 0
    args.best_eval_score_epoch = 0
    args.epochs_since_best_eval_score = 0

    def loss_backward(loss):
        if num_gpus > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if args.grad_accum_steps > 1:
            loss = loss / args.grad_accum_steps
        loss.backward()
        return loss

    if args.do_regression:
        # TODO integrate with textattack `metrics` package
        loss_fct = torch.nn.MSELoss()
    else:
        loss_fct = torch.nn.CrossEntropyLoss()

    for epoch in tqdm.trange(
        int(args.num_train_epochs), desc="Epoch", position=0, leave=True
    ):
        if adversarial_training:
            if epoch >= args.num_clean_epochs:
                if (epoch - args.num_clean_epochs) % args.attack_period == 0:
                    # only generate a new adversarial training set every args.attack_period epochs
                    # after the clean epochs
                    logger.info("Attacking model to generate new training set...")

                    adv_attack_results = _generate_adversarial_examples(
                        model_wrapper, attack_class, list(zip(train_text, train_labels))
                    )
                    adv_train_text = [r.perturbed_text() for r in adv_attack_results]
                    train_dataloader = _make_dataloader(
                        tokenizer, adv_train_text, train_labels, args.batch_size
                    )
            else:
                logger.info(f"Running clean epoch {epoch+1}/{args.num_clean_epochs}")

        prog_bar = tqdm.tqdm(train_dataloader, desc="Iteration", position=0, leave=True)

        # Use these variables to track training accuracy during classification.
        correct_predictions = 0
        total_predictions = 0
        for step, batch in enumerate(prog_bar):
            input_ids, labels = batch
            labels = labels.to(device)
            if isinstance(input_ids, dict):
                ## dataloader collates dict backwards. This is a workaround to get
                # ids in the right shape for HuggingFace models
                input_ids = {
                    k: torch.stack(v).T.to(device) for k, v in input_ids.items()
                }
                logits = model(**input_ids)[0]
            else:

                input_ids = input_ids.to(device)
                logits = model(input_ids)

            if args.do_regression:
                # TODO integrate with textattack `metrics` package
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)
                pred_labels = logits.argmax(dim=-1)
                correct_predictions += (pred_labels == labels).sum().item()
                total_predictions += len(pred_labels)

            loss = loss_backward(loss)
            tr_loss += loss.item()

            if global_step % args.tb_writer_step == 0:
                tb_writer.add_scalar("loss", loss.item(), global_step)
                if scheduler is not None:
                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                else:
                    tb_writer.add_scalar("lr", args.learning_rate, global_step)
            if global_step > 0:
                prog_bar.set_description(f"Loss {tr_loss/global_step}")
            if (step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
            # Save model checkpoint to file.
            if (
                global_step > 0
                and (args.checkpoint_steps > 0)
                and (global_step % args.checkpoint_steps) == 0
            ):
                _save_model_checkpoint(model, args.output_dir, global_step)

            # Inc step counter.
            global_step += 1

        # Print training accuracy, if we're tracking it.
        if total_predictions > 0:
            train_acc = correct_predictions / total_predictions
            logger.info(f"Train accuracy: {train_acc*100}%")
            tb_writer.add_scalar("epoch_train_score", train_acc, epoch)

        # Check accuracy after each epoch.
        # skip args.num_clean_epochs during adversarial training
        if (not adversarial_training) or (epoch >= args.num_clean_epochs):
            eval_score = _get_eval_score(model, eval_dataloader, args.do_regression)
            tb_writer.add_scalar("epoch_eval_score", eval_score, epoch)

            if args.checkpoint_every_epoch:
                _save_model_checkpoint(model, args.output_dir, args.global_step)

            logger.info(
                f"Eval {'pearson correlation' if args.do_regression else 'accuracy'}: {eval_score*100}%"
            )
            if eval_score > args.best_eval_score:
                args.best_eval_score = eval_score
                args.best_eval_score_epoch = epoch
                args.epochs_since_best_eval_score = 0
                _save_model(model, args.output_dir, args.weights_name, args.config_name)
                logger.info(f"Best acc found. Saved model to {args.output_dir}.")
                _save_args(args, args_save_path)
                logger.info(f"Saved updated args to {args_save_path}")
            else:
                args.epochs_since_best_eval_score += 1
                if (args.early_stopping_epochs > 0) and (
                    args.epochs_since_best_eval_score > args.early_stopping_epochs
                ):
                    logger.info(
                        f"Stopping early since it's been {args.early_stopping_epochs} steps since validation acc increased"
                    )
                    break

        if args.check_robustness:
            samples_to_attack = list(zip(train_text, train_labels))
            samples_to_attack = random.sample(samples_to_attack, 1000)
            adv_attack_results = _generate_adversarial_examples(
                model_wrapper, attack_class, samples_to_attack
            )
            attack_types = [r.__class__.__name__ for r in adv_attack_results]
            attack_types = collections.Counter(attack_types)

            adv_acc = 1 - (
                attack_types["SkippedAttackResult"] / len(adv_attack_results)
            )
            total_attacks = (
                attack_types["SuccessfulAttackResult"]
                + attack_types["FailedAttackResult"]
            )
            adv_succ_rate = attack_types["SuccessfulAttackResult"] / total_attacks
            after_attack_acc = attack_types["FailedAttackResult"] / len(
                adv_attack_results
            )

            tb_writer.add_scalar("robustness_test_acc", adv_acc, global_step)
            tb_writer.add_scalar("robustness_total_attacks", total_attacks, global_step)
            tb_writer.add_scalar(
                "robustness_attack_succ_rate", adv_succ_rate, global_step
            )
            tb_writer.add_scalar(
                "robustness_after_attack_acc", after_attack_acc, global_step
            )

            logger.info(f"Eval after-attack accuracy: {100*after_attack_acc}%")

    # read the saved model and report its eval performance
    logger.info("Finished training. Re-loading and evaluating model from disk.")
    model_wrapper = model_from_args(args, args.num_labels)
    model = model_wrapper.model
    model.load_state_dict(torch.load(os.path.join(args.output_dir, args.weights_name)))
    eval_score = _get_eval_score(model, eval_dataloader, args.do_regression)
    logger.info(
        f"Saved model {'pearson correlation' if args.do_regression else 'accuracy'}: {eval_score*100}%"
    )

    if args.save_last:
        _save_model(model, args.output_dir, args.weights_name, args.config_name)

    # end of training, save tokenizer
    try:
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Saved tokenizer {tokenizer} to {args.output_dir}.")
    except AttributeError:
        logger.warn(
            f"Error: could not save tokenizer {tokenizer} to {args.output_dir}."
        )

    # Save a little readme with model info
    write_readme(args, args.best_eval_score, args.best_eval_score_epoch)

    _save_args(args, args_save_path)
    tb_writer.close()
    logger.info(f"Wrote final training args to {args_save_path}.")
