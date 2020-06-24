import json
import logging
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
import tqdm

import textattack
import transformers

from .train_args_helpers import dataset_from_args, model_from_args

device = textattack.shared.utils.device
logger = textattack.shared.logger


def make_directories(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def batch_encode(tokenizer, text_list):
    return [tokenizer.encode(text_input) for text_input in text_list]
    # try:
        # return tokenizer.batch_encode(text_list) # TODO get batch encoding to work with fast tokenizer
    # except AttributeError:
        # return [tokenizer.encode(text_input) for text_input in text_list]


def train_model(args):
    start_time = time.time()
    make_directories(args.output_dir)

    num_gpus = torch.cuda.device_count()

    # Start Tensorboard and log hyperparams.
    from tensorboardX import SummaryWriter

    tb_writer = SummaryWriter(args.output_dir)
    def is_writable_type(obj):
        for ok_type in [bool, int, str, float]:
            if isinstance(obj, ok_type): return True
        return False
    args_dict = {k: v for k,v in vars(args).items() if is_writable_type(v)}
    
    tb_writer.add_hparams(args_dict, {})
    
    # Save logger writes to file
    log_txt_path = os.path.join(args.output_dir, "log.txt")
    fh = logging.FileHandler(log_txt_path)
    fh.setLevel(logging.DEBUG)
    logger.info(f"Writing logs to {log_txt_path}.")
    
    # Save args to file
    args_save_path = os.path.join(args.output_dir, "train_args.json")
    with open(args_save_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(args_dict, indent=2) + "\n")
    logger.info(f"Wrote training args to {args_save_path}.")

    # Use Weights & Biases, if enabled.
    if args.enable_wandb:
        wandb.init(sync_tensorboard=True)

    # Get list of text and list of label (integers) from disk.
    train_text, train_labels, eval_text, eval_labels = dataset_from_args(
        args
    )
    
    # Filter labels
    if args.allowed_labels:
        logger.info(f"Filtering samples with labels outside of {args.allowed_labels}.")
        final_train_text, final_train_labels = [], []
        for text, label in zip(train_text, train_labels):
            if label in args.allowed_labels:
                final_train_text.append(text)
                final_train_labels.append(label)
        logger.info(f"Filtered {len(train_text)} train samples to {len(final_train_text)} points.")
        train_text, train_labels = final_train_text, final_train_labels
        final_eval_text, final_eval_labels = [], []
        for text, label in zip(eval_text, eval_labels):
            if label in args.allowed_labels:
                final_eval_text.append(text)
                final_eval_labels.append(label)
        logger.info(f"Filtered {len(eval_text)} dev samples to {len(final_eval_text)} points.")
        eval_text, eval_labels = final_eval_text, final_eval_labels
    
    label_id_len = len(train_labels)
    label_set = set(train_labels)
    num_labels = len(label_set)
    logger.info(f"Loaded dataset. Found: {num_labels} labels: ({sorted(label_set)})")

    train_examples_len = len(train_text)

    if len(train_labels) != train_examples_len:
        raise ValueError(
            f"Number of train examples ({train_examples_len}) does not match number of labels ({len(train_labels)})"
        )
    if len(eval_labels) != len(eval_text):
        raise ValueError(
            f"Number of teste xamples ({len(eval_text)}) does not match number of labels ({len(eval_labels)})"
        )

    model = model_from_args(args, num_labels)
    tokenizer = model.tokenizer

    logger.info(f"Tokenizing training data. (len: {train_examples_len})")
    train_text_ids = batch_encode(tokenizer, train_text)
    logger.info(f"Tokenizing eval data (len: {len(eval_labels)})")
    eval_text_ids = batch_encode(tokenizer, eval_text)
    load_time = time.time()
    logger.info(f"Loaded data and tokenized in {load_time-start_time}s")
    # print(model)

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

    optimizer = transformers.optimization.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    scheduler = transformers.optimization.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_proportion,
        num_training_steps=num_train_optimization_steps,
    )

    global_step = 0

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

    def get_eval_acc():
        model.eval()
        correct = 0
        total = 0
        for input_ids, labels in tqdm.tqdm(eval_dataloader, desc="Evaluating accuracy"):
            if isinstance(input_ids, dict):
                ## HACK: dataloader collates dict backwards. This is a temporary
                # workaround to get ids in the right shape
                input_ids = {
                    k: torch.stack(v).T.to(device) for k, v in input_ids.items()
                }
            labels = labels.to(device)

            with torch.no_grad():
                logits = textattack.shared.utils.model_predict(model, input_ids)

            correct += (logits.argmax(dim=1) == labels).sum()
            total += len(labels)
        
        model.train()
        return float(correct) / total

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

        logger.info(f"Best acc found. Saved model to {args.output_dir}.")

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
    best_eval_acc = 0
    steps_since_best_eval_acc = 0

    def loss_backward(loss):
        if num_gpus > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if args.grad_accum_steps > 1:
            loss = loss / args.grad_accum_steps
        loss.backward()
        return loss

    for _ in tqdm.trange(int(args.num_train_epochs), desc="Epoch"):
        prog_bar = tqdm.tqdm(train_dataloader, desc="Iteration")
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
            
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
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
            if global_step > 0 and (args.checkpoint_steps > 0) and (global_step % args.checkpoint_steps) == 0:
                save_model_checkpoint()

            model.zero_grad()

            # Inc step counter.
            global_step += 1

        # Check accuracy after each epoch.
        eval_acc = get_eval_acc()
        tb_writer.add_scalar("epoch_eval_acc", eval_acc, global_step)

        if args.checkpoint_every_epoch:
            save_model_checkpoint()

        logger.info(f"Eval acc: {eval_acc*100}%")
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            steps_since_best_eval_acc = 0
            save_model()
        else:
            steps_since_best_eval_acc += 1
            if (args.early_stopping_epochs > 0) and (
                steps_since_best_eval_acc > args.early_stopping_epochs
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
        logger.warn(f"Error: could not save tokenizer {tokenizer} to {args.output_dir}.")
