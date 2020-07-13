"""
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
"""
import sys

from google.protobuf import text_format
import tensorflow as tf

tf.get_logger().setLevel("INFO")


def LoadModel(sess, graph, gd_file, ckpt_file):
    """Load the model from GraphDef and Checkpoint.

    Args:
      gd_file: GraphDef proto text file.
      ckpt_file: TensorFlow Checkpoint file.

    Returns:
      TensorFlow session and tensors dict.
    """
    with graph.as_default():
        sys.stderr.write("Recovering graph.\n")
        with tf.io.gfile.GFile(gd_file) as f:
            s = f.read()
            gd = tf.compat.v1.GraphDef()
            text_format.Merge(s, gd)

        tf.compat.v1.logging.info("Recovering Graph %s", gd_file)
        t = {}
        [
            t["states_init"],
            t["lstm/lstm_0/control_dependency"],
            t["lstm/lstm_1/control_dependency"],
            t["softmax_out"],
            t["class_ids_out"],
            t["class_weights_out"],
            t["log_perplexity_out"],
            t["inputs_in"],
            t["targets_in"],
            t["target_weights_in"],
            t["char_inputs_in"],
            t["all_embs"],
            t["softmax_weights"],
            t["global_step"],
        ] = tf.import_graph_def(
            gd,
            {},
            [
                "states_init",
                "lstm/lstm_0/control_dependency:0",
                "lstm/lstm_1/control_dependency:0",
                "softmax_out:0",
                "class_ids_out:0",
                "class_weights_out:0",
                "log_perplexity_out:0",
                "inputs_in:0",
                "targets_in:0",
                "target_weights_in:0",
                "char_inputs_in:0",
                "all_embs_out:0",
                "Reshape_3:0",
                "global_step:0",
            ],
            name="",
        )

        sys.stderr.write("Recovering checkpoint %s\n" % ckpt_file)
        sess.run("save/restore_all", {"save/Const:0": ckpt_file})
        sess.run(t["states_init"])

    return t
