import numpy as np
import tensorflow as tf


def build_path(prefix, data_type, model_type, num_layers, postpix=""):
    return prefix + data_type + "-" + model_type + "-" + str(num_layers) + postpix


if __name__ == "__main__":
    ap = [
        [
            [
                [1, 1, 1],
                [2, 2, 2]
            ],
            [
                [3, 3, 3],
                [4, 4, 4]
            ]
        ],
        [
            [
                [0.1, 0.1, 0.1],
                [0.2, 0.2, 0.2]
            ],
            [
                [0.3, 0.3, 0.3],
                [0.4, 0.4, 0.4]
            ]
        ]
    ]

    ap = tf.pad(ap, np.array([[0, 0], [0, 0], [3, 3], [0, 0]]), "CONSTANT", name="pad_wide_conv")
    print(ap)
