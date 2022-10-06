import os
import sys

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras

def recover_and_fit(
        model,
        ckpt_dir = './ckpt_dir',
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    ):
    ckpt_path = ckpt_dir + "/cp-{epoch}.ckpt"
    my_callbacks = [
      keras.callbacks.CSVLogger('his.csv', append = True),
      keras.callbacks.ModelCheckpoint(ckpt_path)
    ]
    if callbacks is None:
      callbacks = my_callbacks
    else:
      callbacks = my_callbacks + callbacks
    try:
      his = pd.read_csv('his.csv')
      initial_epoch = his['epoch'].to_numpy()[-1]
      recover_ckpt_path = ckpt_dir + "/cp-{initial_epoch}.ckpt"
      model = keras.models.load_model(recover_ckpt_path)
    except Exception as e:
      print(e, sys.stderr)
      print('Some error happened, train from the scratch.')
    model.fit(
        x,
        y,
        batch_size,
        epochs,
        verbose,
        callbacks,
        validation_split,
        validation_data,
        shuffle,
        class_weight,
        sample_weight,
        initial_epoch,
        steps_per_epoch,
        validation_steps,
        validation_batch_size,
        validation_freq,
        max_queue_size,
        workers,
        use_multiprocessing,
      )