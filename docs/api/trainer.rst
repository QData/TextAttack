Training API Reference
==========================

Trainer
------------
The :class:`~textattack.Trainer` class provides an API for adversarial training with features builtin for standard use cases.
It is designed to be similar to the :obj:`Trainer` class provided by 🤗  Transformers library. 
Custom behaviors can be added by subclassing the class and overriding these methods:

- :meth:`training_step`: Peform a single training step. Override this for custom forward pass or custom loss.
- :meth:`evaluate_step`: Peform a single evaluation step. Override this for custom foward pass.
- :meth:`get_train_dataloader`: Creates the PyTorch DataLoader for training. Override this for custom batch setup.
- :meth:`get_eval_dataloader`: Creates the PyTorch DataLoader for evaluation. Override this for custom batch setup.
- :meth:`get_optimizer_and_scheduler`: Creates the optimizer and scheduler for training. Override this for custom optimizer and scheduler.

The pseudocode for how training is done:

.. code-block::

   train_preds = []
   train_targets = []
   for batch in train_dataloader:
      loss, preds, targets = training_step(model, tokenizer, batch)
      train_preds.append(preds)
      train_targets.append(targets)

      # clear gradients
      optimizer.zero_grad()

      # backward
      loss.backward()

      # update parameters
      optimizer.step()
      if scheduler:
         scheduler.step()

   # Calculate training accuracy using `train_preds` and `train_targets`

   eval_preds = []
   eval_targets = []
   for batch in eval_dataloader:
      loss, preds, targets = training_step(model, tokenizer, batch)
      eval_preds.append(preds)
      eval_targets.append(targets)

   # Calculate eval accuracy using `eval_preds` and `eval_targets`
  

.. autoclass:: textattack.Trainer
   :members:


TrainingArgs
-------------
Training arguments to be passed to :class:`~textattack.Trainer` class.

.. autoclass:: textattack.TrainingArgs
   :members:

