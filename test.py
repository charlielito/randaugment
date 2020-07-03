import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
from randaugment import randaugment

tfds_dataset, tfds_info = tfds.load("kmnist", split="train", with_info=True)
num_images = tfds_info.splits["train"].num_examples
num_classes = tfds_info.features["label"].num_classes
print(num_classes, num_images)


def _preprocess(x):
    x["image"] = tf.image.grayscale_to_rgb(x["image"])
    x["image"] = randaugment(x["image"], 3, 8)
    return x


dataset = tfds_dataset.map(_preprocess).batch(1)

N_SAMPLES = 10
for batch, _ in zip(dataset, range(N_SAMPLES)):
    images = batch["image"]
    print(images.shape)
    # print(images[0, ...].numpy())
    plt.imshow((images[0, ...].numpy() * 255).astype(np.uint8))
    plt.show()
