from eukaryote.t4a.train_support.trainer_textattack import T4A_Trainer
from eukaryote.training_args import TrainingArgs


def train_huggingface(
    model_wrapper,
    dataset_train,
    dataset_eval=None,
    attack=None,
    epochs=None,
    early_stopping_epochs=None,
    learning_rate=None,
    batch_size=None,
    training_args=None,
    after_epoch_fn=None,
    disable_save=False,
    skip_eval=False,
    retain_adversarial_examples=True,
    num_train_examples=None,
):
    """Train a model using the TextAttack Trainer.

    Note: `train_textattack` is exported as `train_huggingface`.

    Args:
        model_wrapper (textattack.models.wrappers.ModelWrapper):
            The model to train as a textattack `ModelWrapper`.
        dataset_train (textattack.datasets.Dataset):
            The dataset to train the model on.
        dataset_eval (Optional[textattack.datasets.Dataset]):
            The dataset to evaluate the model on.
        attack (Optional[textattack.Attack]):
            The attack, if training on adversarial examples.
        epochs (Optional[int]):
            The number of epochs. Defaults to 1.
        early_stopping_epochs (Optional[int]):
            The number of epochs validation must increase before early
            stopping.
        learning_rate (Optional[float]):
            The learning rate. Defaults to 5e-5.
        batch_size (Optional[int]):
            The batch size. Defaults to 8 for training and 32 for evaluation
            when not provided.
        training_args (Optional[dict]):
            Additional kwargs to pass to a `textattack.TrainingArgs`.
        after_epoch_fn (Optional[Callable[[int], Any]]):
            A function to be called back after each epoch.
        disable_save (bool):
            Disable all automatic model saving in TextAttack.
        skip_eval (bool):
            Skips evaluation. Defaults to false.

            Note: early stopping does not work when this is enabled.
        retain_adversarial_examples (bool):
            By default, TextAttack generates adversarial examples every N
            epochs, then throws them away after a single-epoch use. Enabling
            this flag changes that behavior and retains the same adversarial
            dataset until the next one is generated. Defaults to true.
        num_train_examples (Optional[Union[int, float]]):
            Use a random sample of the training dataset as opposed to the
            entire set (which is the native TextAttack behavior). Can be given
            as either an number of examples or a proportion.
    """

    # Use TextAttack defaults if not provided
    training_args = training_args or {}
    if epochs:
        training_args["num_epochs"] = epochs
    if early_stopping_epochs:
        training_args["early_stopping_epochs"] = early_stopping_epochs
    if learning_rate:
        training_args["learning_rate"] = learning_rate
    if batch_size:
        training_args["per_device_train_batch_size"] = batch_size
        training_args["per_device_eval_batch_size"] = batch_size
    if attack:
        if "num_train_adv_examples" not in training_args:
            training_args["num_train_adv_examples"] = len(dataset_train)
        if "num_clean_epochs" not in training_args:
            training_args["num_clean_epochs"] = 0
    training_args = TrainingArgs(**training_args)

    # If no evaluation set is provided, evaluate over the training set
    # This may be desirable if wanting to train over an entire dataset
    # Even when skip_eval is passed, the textattack.Trainer constructor checks
    # there is some eval dataset
    if dataset_eval is None:
        dataset_eval = dataset_train

    trainer = T4A_Trainer(
        model_wrapper,
        attack=attack,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        training_args=training_args,
        after_epoch_fn=after_epoch_fn,
        disable_save=disable_save,
        skip_eval=skip_eval,
        retain_adversarial_examples=retain_adversarial_examples,
        num_train_examples=num_train_examples,
    )

    trainer.train()
