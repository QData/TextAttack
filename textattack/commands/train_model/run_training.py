import json
import logging
import os
import time

import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
import tqdm
import transformers

import textattack

from .train_args_helpers import dataset_from_args, model_from_args, write_readme

device = textattack.shared.utils.device
logger = textattack.shared.logger


def make_directories(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def batch_encode(tokenizer, text_list):
    if hasattr(tokenizer, "batch_encode"):
        return tokenizer.batch_encode(text_list)
    else:
        return [tokenizer.encode(text_input) for text_input in text_list]


def train_model(args):
    logger.warn(
        "WARNING: TextAttack's model training feature is in beta. Please report any issues on our Github page, https://github.com/QData/TextAttack/issues."
    )
    start_time = time.time()
    make_directories(args.output_dir)

    num_gpus = torch.cuda.device_count()

    # Save logger writes to file
    log_txt_path = os.path.join(args.output_dir, "log.txt")
    fh = logging.FileHandler(log_txt_path)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.info(f"Writing logs to {log_txt_path}.")

    # Use Weights & Biases, if enabled.
    if args.enable_wandb:
        global wandb
        import wandb

        wandb.init(sync_tensorboard=True)

    # Get list of text and list of label (integers) from disk.
    train_text, train_labels, eval_text, eval_labels = dataset_from_args(args)

    # Filter labels
    if args.allowed_labels:
        logger.info(f"Filtering samples with labels outside of {args.allowed_labels}.")
        final_train_text, final_train_labels = [], []
        for text, label in zip(train_text, train_labels):
            if label in args.allowed_labels:
                final_train_text.append(text)
                final_train_labels.append(label)
        logger.info(
            f"Filtered {len(train_text)} train samples to {len(final_train_text)} points."
        )
        train_text, train_labels = final_train_text, final_train_labels
        final_eval_text, final_eval_labels = [], []
        for text, label in zip(eval_text, eval_labels):
            if label in args.allowed_labels:
                final_eval_text.append(text)
                final_eval_labels.append(label)
        logger.info(
            f"Filtered {len(eval_text)} dev samples to {len(final_eval_text)} points."
        )
        eval_text, eval_labels = final_eval_text, final_eval_labels

    label_id_len = len(train_labels)
    label_set = set(train_labels)
    args.num_labels = len(label_set)
    logger.info(
        f"Loaded dataset. Found: {args.num_labels} labels: ({sorted(label_set)})"
    )

    if isinstance(train_labels[0], float):
        # TODO come up with a more sophisticated scheme for when to do regression
        logger.warn(f"Detected float labels. Doing regression.")
        args.num_labels = 1
        args.do_regression = True
    else:
        args.do_regression = False

    train_examples_len = len(train_text)

    if len(train_labels) != train_examples_len:
        raise ValueError(
            f"Number of train examples ({train_examples_len}) does not match number of labels ({len(train_labels)})"
        )
    if len(eval_labels) != len(eval_text):
        raise ValueError(
            f"Number of teste xamples ({len(eval_text)}) does not match number of labels ({len(eval_labels)})"
        )

    model = model_from_args(args, args.num_labels)
    tokenizer = model.tokenizer

    logger.info(f"Tokenizing training data. (len: {train_examples_len})")
    train_text_ids = batch_encode(tokenizer, train_text)
    logger.info(f"Tokenizing eval data (len: {len(eval_labels)})")
    eval_text_ids = batch_encode(tokenizer, eval_text)
    load_time = time.time()
    logger.info(f"Loaded data and tokenized in {load_time-start_time}s")

    # multi-gpu training
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    logger.info(f"Training model across {num_gpus} GPUs")

    num_train_optimization_steps = (
        int(train_examples_len / args.batch_size / args.grad_accum_steps)
        * args.num_train_epochs
    )

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

    global_step = 0

    # Start Tensorboard and log hyperparams.
    from tensorboardX import SummaryWriter

    tb_writer = SummaryWriter(args.output_dir)

    def is_writable_type(obj):
        for ok_type in [bool, int, str, float]:
            if isinstance(obj, ok_type):
                return True
        return False

    args_dict = {k: v for k, v in vars(args).items() if is_writable_type(v)}

    tb_writer.add_hparams(args_dict, {})

    # Start training
    logger.info("***** Running training *****")
    logger.info(f"\tNum examples = {train_examples_len}")
    logger.info(f"\tBatch size = {args.batch_size}")
    logger.info(f"\tMax sequence length = {args.max_length}")
    logger.info(f"\tNum steps = {num_train_optimization_steps}")
    logger.info(f"\tNum epochs = {args.num_train_epochs}")
    logger.info(f"\tLearning rate = {args.learning_rate}")

    train_input_ids = np.array(train_text_ids)
    train_labels = np.array(train_labels)
    train_data = list((ids, label) for ids, label in zip(train_input_ids, train_labels))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.batch_size
    )

    eval_input_ids = np.array(eval_text_ids)
    eval_labels = np.array(eval_labels)
    eval_data = list((ids, label) for ids, label in zip(eval_input_ids, eval_labels))
    eval_sampler = RandomSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=args.batch_size
    )

    def get_eval_score():
        model.eval()
        correct = 0
        total = 0
        logits = []
        labels = []
        for input_ids, batch_labels in eval_dataloader:
            if isinstance(input_ids, dict):
                ## HACK: dataloader collates dict backwards. This is a temporary
                # workaround to get ids in the right shape
                input_ids = {
                    k: torch.stack(v).T.to(device) for k, v in input_ids.items()
                }
            batch_labels = batch_labels.to(device)

            with torch.no_grad():
                batch_logits = textattack.shared.utils.model_predict(model, input_ids)

            logits.extend(batch_logits.cpu().squeeze().tolist())
            labels.extend(batch_labels)

        model.train()
        logits = torch.tensor(logits)
        labels = torch.tensor(labels)

        if args.do_regression:
            pearson_correlation, pearson_p_value = scipy.stats.pearsonr(logits, labels)
            return pearson_correlation
        else:
            preds = logits.argmax(dim=1)
            correct = (preds == labels).sum()
            return float(correct) / len(labels)

    def save_model():
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Only save the model itself

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, args.weights_name)
        output_config_file = os.path.join(args.output_dir, args.config_name)

        torch.save(model_to_save.state_dict(), output_model_file)
        try:
            model_to_save.config.to_json_file(output_config_file)
        except AttributeError:
            # no config
            pass

    global_step = 0

    def save_model_checkpoint():
        # Save model checkpoint
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info(f"Checkpoint saved to {output_dir}.")

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

    for epoch in tqdm.trange(
        int(args.num_train_epochs), desc="Epoch", position=0, leave=False
    ):
        prog_bar = tqdm.tqdm(
            train_dataloader, desc="Iteration", position=1, leave=False
        )
        for step, batch in enumerate(prog_bar):
            input_ids, labels = batch
            labels = labels.to(device)
            if isinstance(input_ids, dict):
                ## HACK: dataloader collates dict backwards. This is a temporary
                # workaround to get ids in the right shape
                input_ids = {
                    k: torch.stack(v).T.to(device) for k, v in input_ids.items()
                }
            logits = textattack.shared.utils.model_predict(model, input_ids)

            if args.do_regression:
                # TODO integrate with textattack `metrics` package
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            loss = loss_backward(loss)

            if global_step % args.tb_writer_step == 0:
                tb_writer.add_scalar("loss", loss.item(), global_step)
                tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
            prog_bar.set_description(f"Loss {loss.item()}")
            if (step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            # Save model checkpoint to file.
            if (
                global_step > 0
                and (args.checkpoint_steps > 0)
                and (global_step % args.checkpoint_steps) == 0
            ):
                save_model_checkpoint()

            model.zero_grad()

            # Inc step counter.
            global_step += 1

        # Check accuracy after each epoch.
        eval_score = get_eval_score()
        tb_writer.add_scalar("epoch_eval_score", eval_score, global_step)

        if args.checkpoint_every_epoch:
            save_model_checkpoint()

        logger.info(
            f"Eval {'pearson correlation' if args.do_regression else 'accuracy'}: {eval_score*100}%"
        )
        if eval_score > args.best_eval_score:
            args.best_eval_score = eval_score
            args.best_eval_score_epoch = epoch
            args.epochs_since_best_eval_score = 0
            save_model()
            logger.info(f"Best acc found. Saved model to {args.output_dir}.")
        else:
            args.epochs_since_best_eval_score += 1
            if (args.early_stopping_epochs > 0) and (
                args.epochs_since_best_eval_score > args.early_stopping_epochs
            ):
                logger.info(
                    f"Stopping early since it's been {args.early_stopping_epochs} steps since validation acc increased"
                )
                break

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

    # Save args to file
    args_save_path = os.path.join(args.output_dir, "train_args.json")
    final_args_dict = {k: v for k, v in vars(args).items() if is_writable_type(v)}
    with open(args_save_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(final_args_dict, indent=2) + "\n")
    logger.info(f"Wrote training args to {args_save_path}.")
