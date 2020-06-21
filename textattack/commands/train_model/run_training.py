import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
import tqdm

import textattack

from .train_args_helpers import dataset_from_args, model_from_args

device = textattack.shared.utils.device
logger = textattack.shared.logger


def make_directories(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def encode_batch(tokenizer, text_list):
    try:
        return tokenizer.encode_batch(text_list)
    except AttributeError:
        return [tokenizer.encode(text) for text in text_list]


def train_model(args):
    from transformers.optimization import AdamW, get_linear_schedule_with_warmup

    start_time = time.time()
    make_directories(args.output_dir)

    num_gpus = torch.cuda.device_count()

    # Start Tensorboard and log hyperparams.
    from tensorboardX import SummaryWriter

    tb_writer = SummaryWriter(args.output_dir)
    args_dict = vars(args)
    del args_dict["func"]
    tb_writer.add_hparams(args_dict, {})

    # Use Weights & Biases, if enabled.
    if args.enable_wandb:
        wandb.init(sync_tensorboard=True)

    # Get list of text and list of label (integers) from disk.
    train_text, train_label_id_list, eval_text, eval_label_id_list = dataset_from_args(
        args
    )
    label_id_len = len(train_label_id_list)
    num_labels = len(set(train_label_id_list))
    logger.info("Loaded dataset. num_labels: %s", num_labels)

    train_examples_len = len(train_text)

    if len(train_label_id_list) != train_examples_len:
        raise ValueError(
            f"Number of train examples ({train_examples_len}) does not match number of labels ({len(train_label_id_list)})"
        )
    if len(eval_label_id_list) != len(eval_text):
        raise ValueError(
            f"Number of teste xamples ({len(eval_text)}) does not match number of labels ({len(eval_label_id_list)})"
        )

    model = model_from_args(args, num_labels)

    logger.info(f"Tokenizing training data. (len: {train_examples_len})")
    train_text_ids = encode_batch(model.tokenizer, train_text)
    logger.info(f"Tokenizing test data (len: {len(eval_label_id_list)})")
    eval_text_ids = encode_batch(model.tokenizer, eval_text)
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

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_proportion,
        num_training_steps=num_train_optimization_steps,
    )

    global_step = 0

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_examples_len)
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Max sequence length = %d", args.max_length)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    logger.info("  Num epochs = %d", args.num_train_epochs)

    train_input_ids = np.array(train_text_ids)
    train_labels = np.array(train_label_id_list)
    train_data = list((ids, label) for ids, label in zip(train_input_ids, train_labels))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.batch_size
    )

    eval_input_ids = np.array(eval_text_ids)
    eval_labels = np.array(eval_label_id_list)
    eval_data = list((ids, label) for ids, label in zip(eval_input_ids, eval_labels))
    eval_sampler = RandomSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=args.batch_size
    )

    def get_eval_acc():
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
            if global_step % args.tb_writer_step == 0:
                tb_writer.add_scalar("loss", loss, global_step)
                tb_writer.add_scalar("lr", loss, global_step)
            loss_backward(loss)
            prog_bar.set_description(f"Loss {loss.item()}")
            if (step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            # Save model checkpoint to file.
            if global_step > 0 and global_step % args.checkpoint_steps == 0:
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
