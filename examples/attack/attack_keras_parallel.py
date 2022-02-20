"""
Recent upgrade of keras versions in TF 2.5+, keras has been moved to tf.keras
This has resulted in certain exceptions when keras models are attacked in parallel
This script fixes this behavior by adding an official hotfix for this situation detailed here:
https://github.com/tensorflow/tensorflow/issues/34697
All models/dataset are similar to keras attack tutorial at :
https://textattack.readthedocs.io/en/latest/2notebook/Example_3_Keras.html#
NOTE: This fix might be deprecated in future TF releases
NOTE: This script is not designed to run in a Jupyter notebook due to conflicting namespace issues
We recommend running it as a script only
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils
import torch

from textattack import AttackArgs, Attacker
from textattack.attack_recipes import PWWSRen2019
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import ModelWrapper

NUM_WORDS = 1000


def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(training_config)
        )
    restored_model.set_weights(weights)
    return restored_model


# Hotfix function
def make_keras_picklable():
    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__


# Run the function
make_keras_picklable()


def transform(x):
    x_transform = []
    for i, word_indices in enumerate(x):
        BoW_array = np.zeros((NUM_WORDS,))
        for index in word_indices:
            if index < len(BoW_array):
                BoW_array[index] += 1
        x_transform.append(BoW_array)
    return np.array(x_transform)


class CustomKerasModelWrapper(ModelWrapper):
    def __init__(self, model):
        self.model = model

    def __call__(self, text_input_list):

        x_transform = []
        for i, review in enumerate(text_input_list):
            tokens = [x.strip(",") for x in review.split()]
            BoW_array = np.zeros((NUM_WORDS,))
            for word in tokens:
                if word in vocabulary:
                    if vocabulary[word] < len(BoW_array):
                        BoW_array[vocabulary[word]] += 1
            x_transform.append(BoW_array)
        x_transform = np.array(x_transform)
        prediction = self.model.predict(x_transform)
        return prediction


model = Sequential()
model.add(Dense(512, activation="relu", input_dim=NUM_WORDS))
model.add(Dropout(0.3))
model.add(Dense(100, activation="relu"))
model.add(Dense(2, activation="sigmoid"))
opt = tf.keras.optimizers.Adam(learning_rate=0.00001)

model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])


(x_train_tokens, y_train), (x_test_tokens, y_test) = tf.keras.datasets.imdb.load_data(
    path="imdb.npz",
    num_words=NUM_WORDS,
    skip_top=0,
    maxlen=None,
    seed=113,
    start_char=1,
    oov_char=2,
    index_from=3,
)


index = int(0.9 * len(x_train_tokens))
x_train = transform(x_train_tokens)[:index]
x_test = transform(x_test_tokens)[index:]
y_train = np.array(y_train[:index])
y_test = np.array(y_test[index:])
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

vocabulary = tf.keras.datasets.imdb.get_word_index(path="imdb_word_index.json")

results = model.fit(
    x_train, y_train, epochs=1, batch_size=512, validation_data=(x_test, y_test)
)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    model_wrapper = CustomKerasModelWrapper(model)
    dataset = HuggingFaceDataset("rotten_tomatoes", None, "test", shuffle=True)

    attack = PWWSRen2019.build(model_wrapper)

    attack_args = AttackArgs(
        num_examples=10,
        checkpoint_dir="checkpoints",
        parallel=True,
        num_workers_per_device=2,
    )

    attacker = Attacker(attack, dataset, attack_args)

    attacker.attack_dataset()
