import tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.layer_utils import count_params
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Dense, \
    Activation, Add, concatenate, SeparableConv2D

from tensorflow.python.keras.callbacks import EarlyStopping, TerminateOnNaN
import matplotlib.pyplot as plt
import time
import os
import pickle
import logging
import math
import copy
import argparse
from tensorflow.python.client import device_lib

import tensorflow_addons as tfa
from keras.callbacks import Callback
from keras import backend as K

import tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.layer_utils import count_params
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Dense, \
    Activation, Add, concatenate, SeparableConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.keras.callbacks import EarlyStopping, TerminateOnNaN
import matplotlib.pyplot as plt
import time
import os
import pickle
import logging
import math
import copy
import tensorflow_addons as tfa

import subprocess
import sys
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


print("installing networkx")
install("networkx")
print("Installed ...")

import networkx as nx

def FusedMBConv(prev, t, filters, strides, act):
    x = tf.keras.layers.Conv2D(t * filters, (3, 3), strides=strides, padding='same')(prev)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(act)(x)

    x = tf.keras.layers.Conv2D(filters, (1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(act)(x)

    if x.shape[1:4] == prev.shape[1:4]:
        x = tf.keras.layers.Add()([x, prev])
    return x


def MBConv(prev, t, filters, strides, act):
    x = tf.keras.layers.Conv2D(t * filters, (1, 1), padding='same')(prev)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(act)(x)

    x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=strides, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(act)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(filters, (1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(act)(x)

    if x.shape[1:4] == prev.shape[1:4]:
        x = tf.keras.layers.Add()([x, prev])
    return x


def efficientnetv2_small(act):
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(24, (3, 3), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(act)(x)

    x = FusedMBConv(x, 1, 24, strides=(1, 1), act=act)
    x = FusedMBConv(x, 1, 24, strides=(1, 1), act=act)

    x = FusedMBConv(x, 4, 48, strides=(2, 2), act=act)
    x = FusedMBConv(x, 4, 48, strides=(1, 1), act=act)
    x = FusedMBConv(x, 4, 48, strides=(1, 1), act=act)
    x = FusedMBConv(x, 4, 48, strides=(1, 1), act=act)

    x = FusedMBConv(x, 4, 64, strides=(2, 2), act=act)
    x = FusedMBConv(x, 4, 64, strides=(1, 1), act=act)
    x = FusedMBConv(x, 4, 64, strides=(1, 1), act=act)
    x = FusedMBConv(x, 4, 64, strides=(1, 1), act=act)

    x = MBConv(x, 4, 128, strides=(2, 2), act=act)
    x = MBConv(x, 4, 128, strides=(1, 1), act=act)
    x = MBConv(x, 4, 128, strides=(1, 1), act=act)
    x = MBConv(x, 4, 128, strides=(1, 1), act=act)
    x = MBConv(x, 4, 128, strides=(1, 1), act=act)
    x = MBConv(x, 4, 128, strides=(1, 1), act=act)

    x = MBConv(x, 6, 160, strides=(1, 1), act=act)
    x = MBConv(x, 6, 160, strides=(1, 1), act=act)
    x = MBConv(x, 6, 160, strides=(1, 1), act=act)
    x = MBConv(x, 6, 160, strides=(1, 1), act=act)
    x = MBConv(x, 6, 160, strides=(1, 1), act=act)
    x = MBConv(x, 6, 160, strides=(1, 1), act=act)
    x = MBConv(x, 6, 160, strides=(1, 1), act=act)
    x = MBConv(x, 6, 160, strides=(1, 1), act=act)
    x = MBConv(x, 6, 160, strides=(1, 1), act=act)

    x = tf.keras.layers.GlobalAvgPool2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

train_ind_half = list(range(0, 20000))
train_ind_full = list(range(0, 40000))
val_ind = list(range(40000, 50000))

AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128  # 512
IMG_SHAPE = 32

threshold = 0.25


def preprocess_image(image, label):
    image = tf.image.resize(image, (IMG_SHAPE, IMG_SHAPE))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image / 255.
    return image, label


trainloader = tf.data.Dataset.from_tensor_slices(
    (x_train[train_ind_full], tf.keras.utils.to_categorical(y_train[train_ind_full])))
trainloader = (
    trainloader
        .map(preprocess_image, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
)

testloader = tf.data.Dataset.from_tensor_slices((x_train[val_ind], tf.keras.utils.to_categorical(y_train[val_ind])))
testloader = (
    testloader
        .map(preprocess_image, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
)


class FiveEpochCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= 6:
            if logs['accuracy'] < threshold:  # 30
                self.model.stop_training = True
        # elif epoch >= 10:
        #    if logs['accuracy'] < 32.5:
        #        self.model.stop_training = True


def fitness_function_effnet(loss):
    start = time.time()

    model = efficientnetv2_small("swish")
    optimizer = tfa.optimizers.AdamW(weight_decay=1e-7)

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    history = model.fit(trainloader.repeat(), epochs=55, steps_per_epoch=40, validation_data=testloader,
                        callbacks=[TerminateOnNaN(), FiveEpochCallback()],
                        verbose=0)

    f = np.nanmax(history.history['val_accuracy'])
    v = np.nanmax(history.history['val_loss'])
    if np.nanmax(history.history['accuracy']) < threshold:
        return None
    finish = time.time()
    total = finish - start
    return f, v, total, history.history


class Node:

    def __init__(self, un_bin_perc, ID):
        self.un_bin_perc = un_bin_perc
        self.op_un_bin = np.random.choice(range(0, 2), p=un_bin_perc)
        self.un = np.random.choice(range(0, 24))
        self.bin = np.random.choice(range(0, 7))
        self.id = ID

    def call(self, in1, in2, name1, name2):

        eps = tf.keras.backend.epsilon()
        if self.op_un_bin == 0:  # unary
            if self.un == 0:
                total = - in1
                msg = "-({})".format(name1)
            elif self.un == 1:
                total = tf.math.log(tf.math.abs(in1) + eps)
                msg = "ln(|{}|)".format(name1)
            elif self.un == 2:
                total = tf.math.divide(tf.math.log(tf.math.abs(in1) + eps), tf.math.log(10.0))
                msg = "log(|{}|)".format(name1)
            elif self.un == 3:
                total = tf.math.exp(in1)
                msg = "exp({})".format(name1)
            elif self.un == 4:
                total = tf.math.abs(in1)
                msg = "|{}|".format(name1)
            elif self.un == 5:
                total = tf.math.divide(1, (1 + tf.math.exp(-in1)))
                msg = "1/(1+exp(-{})".format(name1)
            elif self.un == 6:
                total = tf.math.divide(1, (1 + tf.math.abs(in1)))
                msg = "1/(1+|{}|)".format(name1)
            elif self.un == 7:
                total = tf.math.log(tf.math.abs(1 + tf.math.exp(in1)) + eps)
                msg = "ln(|1+exp({})|)".format(name1)
            elif self.un == 8:
                total = tf.math.erf(in1)
                msg = "erf({})".format(name1)
            elif self.un == 9:
                total = tf.math.erfc(in1)
                msg = "erfc({})".format(name1)
            elif self.un == 10:
                total = tf.math.sin(in1)
                msg = "sin({})".format(name1)
            elif self.un == 11:
                total = tf.math.sinh(in1)
                msg = "sinh({})".format(name1)
            elif self.un == 12:
                total = tf.math.asinh(in1)
                msg = "arcsinh({})".format(name1)
            elif self.un == 13:
                total = tf.math.tanh(in1)
                msg = "tanh({})".format(name1)
            elif self.un == 14:
                total = tf.math.atan(in1)
                msg = "arctan({})".format(name1)
            elif self.un == 15:
                total = tf.math.divide(1, (in1 + eps))
                msg = "1/({})".format(name1)
            elif self.un == 16:
                total = tf.math.bessel_i0(in1)
                msg = "bessel_io({})".format(name1)
            elif self.un == 17:
                total = tf.math.bessel_i0e(in1)
                msg = "bessel_ioe({})".format(name1)
            elif self.un == 18:
                total = tf.math.bessel_i1(in1)
                msg = "bessel_i1({})".format(name1)
            elif self.un == 19:
                total = tf.math.bessel_i1e(in1)
                msg = "bessel_i1e({})".format(name1)
            elif self.un == 20:
                total = tf.math.maximum(in1, 0)
                msg = "max({}, 0)".format(name1)
            elif self.un == 21:
                total = tf.math.minimum(in1, 0)
                msg = "min({}, 0)".format(name1)
            elif self.un == 22:
                total = tf.math.pow(in1, 2)
                msg = "({})^2".format(name1)
            elif self.un == 23:
                total = tf.math.sqrt(tf.math.abs(in1))
                msg = "sqrt({})".format(name1)
            else:
                total = in1
                msg = "un_error"
                print("UNARY: ERROR")
        else:  # binary
            if self.bin == 0:
                total = in1 + in2
                msg = "({})+({})".format(name1, name2)
            elif self.bin == 1:
                total = in1 - in2
                msg = "({})-({})".format(name1, name2)
            elif self.bin == 2:
                total = in1 * in2
                msg = "({})*({})".format(name1, name2)
            elif self.bin == 3:
                total = tf.math.maximum(in1, in2)
                msg = "max({}, {})".format(name1, name2)
            elif self.bin == 4:
                total = tf.math.minimum(in1, in2)
                msg = "min({}, {})".format(name1, name2)
            elif self.bin == 5:
                total = tf.math.divide(in1, (in2 + eps))
                msg = "({}) / ({})".format(name1, name2)
            elif self.bin == 6:
                total = tf.math.divide(in1, tf.math.sqrt(1 + tf.math.pow(in2, 2)))
                msg = "({}) / (sqrt(1+({})^2))".format(name1, name2)
            else:
                print("BINARY: ERROR")
                total = in1
                msg = "bin_err"

        return total, msg


class Loss:

    def __init__(self, label_smoothing=0.05):
        self.nodes = []
        self.root = None
        self.flip = False
        self.age = 0
        self.un_bin_percs = [
            [0.7, 0.3],
            [0.7, 0.3],
            [0.7, 0.3],
            [0.7, 0.3],
        ]
        self.root_un_bin_perc = [0.20, 0.80]
        self.msg = None
        self.adj = None
        self.active_nodes = None
        self.label_smoothing = label_smoothing
        self.phenotype = None
        self.threshold = 0.10
        self.setup()

    def setup(self):
        while True:
            self.nodes = []
            self.adj = np.zeros(shape=(9, 9))
            for i in range(0, 4):
                self.nodes.append(Node(un_bin_perc=self.un_bin_percs[i], ID=i))
                if self.nodes[-1].op_un_bin == 0:  # unary
                    idx = np.random.choice(np.concatenate((np.arange(0, i + 4), np.arange(i + 4 + 1, 4 + 4))))
                    self.adj[i + 4, idx] = 1
                else:  # binary
                    idx = np.random.choice(np.concatenate((np.arange(0, i + 4), np.arange(i + 4 + 1, 4 + 4))))
                    self.adj[i + 4, idx] = 1
                    if idx < i + 4:
                        idx = np.random.choice(
                            np.concatenate((np.arange(0, idx), np.arange(idx + 1, i + 4), np.arange(i + 4 + 1, 4 + 4))))
                    else:
                        idx = np.random.choice(
                            np.concatenate((np.arange(0, i + 4), np.arange(i + 4 + 1, idx), np.arange(idx + 1, 4 + 4))))
                    self.adj[i + 4, idx] = 2

            self.root = Node(un_bin_perc=self.root_un_bin_perc, ID=4)
            if self.root.op_un_bin == 0:  # unary
                idx = np.random.choice(np.arange(4, 4 + 4))
                self.adj[-1, idx] = 1
            else:  # binary
                idx = np.random.choice(np.arange(4, 4 + 4))
                self.adj[-1, idx] = 1
                idx = np.random.choice(np.concatenate((np.arange(4, idx), np.arange(idx + 1, 4 + 4))))
                self.adj[-1, idx] = 2

            msg = self.print_nodes()
            self.set_active()
            active_adj = self.adj[self.active_nodes >= 1][:, self.active_nodes >= 1]
            g = nx.from_numpy_array(active_adj, create_using=nx.DiGraph)
            try:
                nx.find_cycle(g)
                continue
            except:
                pass
            if self.active_nodes[0] == 0 or self.active_nodes[1] == 0:  # check for arg to contain atleast y and y hat
                continue
            if not self.check_integrity():
                continue
            return

    def mutate(self, msgs):
        r = np.random.uniform(0, 1)
        idx = np.random.choice(np.where(self.active_nodes[4:] >= 1)[0].tolist())
        if idx == 4:  # root
            node = self.root
        else:
            node = self.nodes[idx]

        if r >= 0.30:  # change op
            if node.op_un_bin == 0:  # unary
                node.un = np.random.choice(np.concatenate((np.arange(0, node.un), np.arange(node.un + 1, 24))))
            else:
                node.bin = np.random.choice(np.concatenate((np.arange(0, node.bin), np.arange(node.bin + 1, 7))))
        elif r >= 0.15:  # change conn
            c = np.random.choice(np.where(self.adj[idx + 4] == 0)[0])
            if node.op_un_bin == 0:  # unary
                self.adj[idx + 4][self.adj[idx + 4] == 1] = 0
                self.adj[idx + 4][c] = 1
            else:
                if np.random.uniform(0, 1) <= 0.20:  # swap conn
                    idx1 = np.where(self.adj[idx + 4] == 1)[0][0]
                    idx2 = np.where(self.adj[idx + 4] == 2)[0][0]
                    self.adj[idx + 4][idx1] = 2
                    self.adj[idx + 4][idx2] = 1
                else:
                    if np.random.uniform(0, 1) < 0.5:  # change 1st conn
                        self.adj[idx + 4][self.adj[idx + 4] == 1] = 0
                        self.adj[idx + 4][c] = 1
                    else:  # change 2nd conn
                        self.adj[idx + 4][self.adj[idx + 4] == 2] = 0
                        self.adj[idx + 4][c] = 2
        else:  # change un->bin visa versa
            c = np.random.choice(np.where(self.adj[idx + 4] == 0)[0])
            if node.op_un_bin == 0:  # unary -> binary
                node.op_un_bin = 1
                if np.random.uniform(0, 1) <= 0.5:  # add right conn
                    self.adj[idx + 4][c] = 2
                else:
                    self.adj[idx + 4][self.adj[idx + 4] == 1] = 2
                    self.adj[idx + 4][c] = 1
            else:  # binary -> unary
                node.op_un_bin = 0
                if np.random.uniform(0, 1) <= 0.5:  # delete right conn
                    self.adj[idx + 4][self.adj[idx + 4] == 2] = 0
                else:
                    self.adj[idx + 4][self.adj[idx + 4] == 1] = 0
                    self.adj[idx + 4][self.adj[idx + 4] == 2] = 1

        self.set_active()
        active_adj = self.adj[self.active_nodes >= 1][:, self.active_nodes >= 1]
        g = nx.from_numpy_array(active_adj, create_using=nx.DiGraph)
        try:
            nx.find_cycle(g)
            return False
        except:
            pass
        if self.active_nodes[0] == 0 or self.active_nodes[1] == 0:  # check for arg to contain atleast y and y hat
            return False
        if not self.check_integrity():
            return False

        if self.msg in msgs:
            return False

        return True

    def print_nodes(self):
        idx = np.where(self.adj[-1] >= 1)[0]
        if self.root.op_un_bin == 0:
            msg = "root: unary(node{})".format(idx[0] - 4)
        else:
            idx1 = np.where(self.adj[-1] == 1)[0][0]
            idx2 = np.where(self.adj[-1] == 2)[0][0]
            msg = "root: bin(node{}, node{})".format(idx1 - 4, idx2 - 4)
        for i in [3, 2, 1, 0]:
            msg = msg + "\n"
            idx = np.where(self.adj[4 + i] >= 1)[0]
            if self.nodes[i].op_un_bin == 0:  # unary
                if idx >= 4:
                    msg = msg + "node{}: unary(node{})".format(i, idx[0] - 4)
                else:
                    msg = msg + "node{}: unary({})".format(i, idx[0])
            else:  # binary
                idx1 = np.where(self.adj[4 + i] == 1)[0][0]
                idx2 = np.where(self.adj[4 + i] == 2)[0][0]
                if idx1 >= 4 and idx2 >= 4:
                    msg = msg + "node{}: binary(node{}, node{})".format(i, idx1 - 4, idx2 - 4)
                elif idx1 >= 4:
                    msg = msg + "node{}: binary(node{}, {})".format(i, idx1 - 4, idx2)
                elif idx2 >= 4:
                    msg = msg + "node{}: binary({}, node{})".format(i, idx1, idx2 - 4)
                else:
                    msg = msg + "node{}: binary({}, {})".format(i, idx1, idx2)
        return msg + "\n"

    def set_active(self):
        self.root.active = True
        queue = []
        visited = [10]
        queue = np.concatenate((queue, np.where(self.adj[-1] != 0)[0].flatten())).tolist()
        self.active_nodes = np.zeros(shape=(9,))
        self.active_nodes[-1] = 1
        while queue:
            node = int(queue.pop(0))
            if node in visited:
                continue
            visited.append(node)
            queue = np.concatenate((queue, np.where(self.adj[node] != 0)[0].flatten())).tolist()
            self.active_nodes[node] = 1
        self.active_nodes = np.asarray(self.active_nodes, dtype=int)
        return

    @staticmethod
    def normalize(x):
        if np.abs(x.max() - x.min()) < 1e-4:
            return x
        return (x - x.min()) / (x.max() - x.min())

    def check_integrity(self):
        eps = 1e-7
        yhat = tf.convert_to_tensor(
            np.vstack((np.linspace(0 + eps, 1 - eps, 1000), 1 - np.linspace(0 + eps, 1 - eps, 1000))).T,
            dtype=tf.float32)
        y = tf.convert_to_tensor(np.asarray([[0, 1] * 1000]).reshape(1000, 2), dtype=tf.float32)
        p = self.call(y, yhat, pred=False).numpy()
        self.phenotype = self.normalize(p)
        p2 = self.call(y, yhat, cross_entropy=True).numpy()
        p2 = self.normalize(p2)
        if tf.norm(self.normalize(p) - p2) <= 4 * self.threshold or tf.norm(
                self.normalize(-p) - p2) <= 4 * self.threshold:  # too much like cross entropy
            return False
        if np.argmin(p) in range(490, 510) or np.argmin(-p) in range(490, 510):  # min is to close to 0.5
            return False

        diff = np.diff(p)
        if np.all(diff < 0):
            self.flip = True
            self.phenotype = self.normalize(- p)
            return True
        elif np.all(diff > 0):
            self.flip = False
            return True
        else:
            t = np.sign(diff)
            sz = t == 0  # any zeros
            if np.sum(sz) > 10:  # get rid of platues
                return False
            if np.all(t == 0):  # all zeros
                return False
            while sz.any():
                t[sz] = np.roll(t, 1)[sz]
                sz = t == 0
            if np.sum((np.roll(t, 1) - t) != 0) > 2:  # too many oscillations
                return False
            else:
                if t[2] == -1:
                    self.flip = False
                else:  # parabola is upside down, so flip
                    self.flip = True
                    self.phenotype = self.normalize(-p)
                return True

    def call(self, y_true, yhat, cross_entropy=False, pred=True):

        label_smoothing = tf.convert_to_tensor(self.label_smoothing, dtype=yhat.dtype)
        num_classes = tf.cast(tf.shape(y_true)[-1], yhat.dtype)
        y = y_true * (1.0 - label_smoothing) + (
                label_smoothing / num_classes
        )

        if cross_entropy:
            return -tf.math.reduce_sum(y * tf.math.log(yhat), axis=-1)

        one = tf.constant(1.0, dtype=y.dtype)
        neg_one = tf.constant(- 1.0, dtype=y.dtype)

        res = [y, yhat, one, neg_one] + [None] * 4
        msgs = ["y", "yhat", "1", "-1"] + [None] * 4

        while np.any([j == None for j in res]):
            for i in range(4, 8):
                if res[i] is not None:
                    continue
                if self.active_nodes[i] != 1:
                    res[i] = -1
                    continue
                inds = np.where(self.adj[i] >= 1)[0]
                if len(inds) == 2:
                    if res[inds[0]] is None or res[inds[1]] is None:
                        continue
                    idx1 = np.where(self.adj[i] == 1)[0][0]
                    idx2 = np.where(self.adj[i] == 2)[0][0]
                    t, msg = self.nodes[i - 4].call(res[idx1], res[idx2], msgs[idx1], msgs[idx2])
                else:
                    if res[inds[0]] is None:
                        continue
                    t, msg = self.nodes[i - 4].call(res[inds[0]], None, msgs[inds[0]], None)
                res[i] = t
                msgs[i] = msg

        inds = np.where(self.adj[-1] >= 1)[0]
        if len(inds) == 2:
            idx1 = np.where(self.adj[-1] == 1)[0][0]
            idx2 = np.where(self.adj[-1] == 2)[0][0]
            t, msg = self.root.call(res[idx1], res[idx2], msgs[idx1], msgs[idx2])
        else:
            t, msg = self.root.call(res[inds[0]], None, msgs[inds[0]], None)

        self.msg = msg
        if self.flip:
            if pred:
                return -tf.reduce_mean(tf.reduce_sum(t, axis=-1))
            else:
                return -tf.reduce_sum(t, axis=-1)
        if pred:
            return tf.reduce_mean(tf.reduce_sum(t, axis=-1))
        else:
            return tf.reduce_sum(t, axis=-1)


class GeneticAlgorithmRegularized:

    def __init__(self, gen_size):
        self.gen_size = gen_size
        self.gen = []
        self.fitness = []
        self.prev_individuals = []
        self.best_individuals = []
        self.best_fit = []
        self.mean_fit = []
        self.median_fit = []
        self.min_fit = []
        self.similarities = []
        self.phenotypes = []
        self.functions = []
        self.index = self.gen_size

        self.init_gen = []
        self.init_fitness = []

    def initialize(self, fitness_function, init_size):
        msg = "TRAINING INITIAL POPULATION"
        print(msg)
        logging.info(msg)
        start = time.time()
        self.init_gen = []
        self.init_fitness = []
        for i in range(0, init_size):
            self.init_gen.append(Loss())  # random loss
            result = fitness_function(self.init_gen[i].call)
            if result is None:
                msg = " INIT MODEL ARCHITECTURE FAILED..."
                self.init_fitness.append(0)
            else:
                f, v, t, hist = result
                self.init_gen[i].hist = hist
                msg = " MODEL {} -> Val Acc: {}, Val Loss: {}, Time: {}, Fun: {}".format(i, f, v, t,
                                                                                         self.init_gen[i].msg)
                self.init_fitness.append(f)
            print(msg)
            logging.info(msg)

        self.init_fitness = np.asarray(self.init_fitness)
        self.init_gen = np.asarray(self.init_gen)
        bst = np.argsort(-self.init_fitness)[0:self.gen_size]
        self.fitness = self.init_fitness[bst]
        self.gen = self.init_gen[bst]

        finish = time.time()
        msg = " Time Elapsed: {} min".format((finish - start) / 60.0)
        print(msg)
        logging.info(msg)

    def evolve(self, max_iter, fitness_function):
        start = time.time()
        for k in range(self.index, self.index + max_iter):

            inds = np.random.choice(range(0, self.gen_size), 15)
            fits = self.fitness[inds]
            bst = np.argmax(fits)
            chosen = self.gen[inds[bst]]
            max_redo = 3
            f = 0
            msg = " -> CHILD ARCHITECTURE FAILED"
            while max_redo > 0:
                done_mutating = 500
                while done_mutating > 0:
                    child = copy.deepcopy(chosen)
                    child.age = 0
                    if child.mutate([ind.msg for ind in self.gen]):
                        break
                    else:
                        msg = " -> FAILED MUTATION"
                    done_mutating -= 1
                result = fitness_function(child.call)
                if result is None:
                    max_redo -= 1
                    if max_redo == 0:
                        f = 0
                        child.hist = None
                else:
                    f, v, t, hist = result
                    child.hist = hist
                    msg = " -> CHILD {}-> Val Acc: {}, Val Loss: {}, Time: {}, Fun: {}".format(k, f, v, t, child.msg)
                    break
            print(msg)
            logging.info(msg)

            for i in range(0, self.gen_size):
                self.gen[i].age += 1

            if f == 0:
                fit_mean = np.mean(self.fitness)
                fit_median = np.median(self.fitness)
                fit_min = np.min(self.fitness)
                fit_best = np.max(self.fitness)
                self.best_fit.append(fit_best)
                self.mean_fit.append(fit_mean)
                self.median_fit.append(fit_median)
                self.min_fit.append(fit_min)
                self.prev_individuals.append(child)
                continue

            ages = np.asarray([ind.age for ind in self.gen])
            oldest = np.max(ages)
            oldest = np.where(ages == oldest)[0]
            worst = np.argmin(self.fitness[oldest])
            self.prev_individuals.append(self.gen[oldest[worst]])
            self.fitness[oldest[worst]] = f
            self.gen[oldest[worst]] = child

            fit_mean = np.mean(self.fitness)
            fit_median = np.median(self.fitness)
            fit_min = np.min(self.fitness)
            fit_best = np.max(self.fitness)
            self.best_fit.append(fit_best)
            self.mean_fit.append(fit_mean)
            self.median_fit.append(fit_median)
            self.min_fit.append(fit_min)

            msg = "  Best Fit: {}, Mean Fit: {}, Median Fit: {}, Min Fit: {}".format(fit_best, fit_mean, fit_median,
                                                                                     fit_min)
            print(msg)
            logging.info(msg)

        finish = time.time()

        msg = " Time Elapsed: {} min".format((finish - start) / 60.0)
        print(msg)
        logging.info(msg)
        self.index += max_iter


def fitness_function_test(ind):
    if np.random.uniform(0, 1) >= 0.25:
        return None
    eps = 1e-7
    yhat = tf.convert_to_tensor(
        np.vstack((np.linspace(0 + eps, 1 - eps, 1000), 1 - np.linspace(0 + eps, 1 - eps, 1000))).T,
        dtype=tf.float32)
    y = tf.convert_to_tensor(np.asarray([[0, 1] * 1000]).reshape(1000, 2), dtype=tf.float32)
    return ind(y, yhat), 1, 1, None


def create_parser():
    '''
    Create a command line parser for the XOR experiment
    '''
    parser = argparse.ArgumentParser(description='Neural Loss Evolution')
    parser.add_argument('--logs_file', type=str,
                        default='nas_loss_regularized_effnet_5000_init_proj_test2_categorical_15.log',
                        help='Output File For Logging')
    parser.add_argument('--save_dir', type=str, default='nas_loss_regularized_effnet_5000_init',
                        help='Save Directory for saving Logs/Checkups')
    parser.add_argument('--save_file', type=str, default='nas_loss_regularized_algo_5000_init',
                        help='Save File for Algorithm')
    parser.add_argument('--batch', type=int, default=0, help="Batch")

    return parser


def lr_warmup_cosine_decay(global_step,
                           warmup_steps,
                           hold=0,
                           total_steps=0,
                           start_lr=0.0,
                           target_lr=1e-3):
    # Cosine decay
    learning_rate = 0.5 * target_lr * (
            1 + np.cos(np.pi * (global_step - warmup_steps - hold) / float(total_steps - warmup_steps - hold)))

    # Target LR * progress of warmup (=1 at the final warmup step)
    warmup_lr = target_lr * (global_step / warmup_steps)

    # Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether `global_step < warmup_steps` and we're still holding.
    # i.e. warm up if we're still warming up and use cosine decayed lr otherwise
    if hold > 0:
        learning_rate = np.where(global_step > warmup_steps + hold,
                                 learning_rate, target_lr)

    learning_rate = np.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return learning_rate


class WarmupCosineDecay(tf.keras.callbacks.Callback):
    def __init__(self, total_steps=0, warmup_steps=0, start_lr=0.0, target_lr=1e-3, hold=0):
        super(WarmupCosineDecay, self).__init__()
        self.start_lr = start_lr
        self.hold = hold
        self.total_steps = total_steps
        self.global_step = 0
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.lrs = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = self.model.optimizer.lr.numpy()
        self.lrs.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = lr_warmup_cosine_decay(global_step=self.global_step,
                                    total_steps=self.total_steps,
                                    warmup_steps=self.warmup_steps,
                                    start_lr=self.start_lr,
                                    target_lr=self.target_lr,
                                    hold=self.hold)
        K.set_value(self.model.optimizer.lr, lr + 1e-7)
        # tf.print(self.model.optimizer.lr)


def fitness_function_effnet2(loss):
    start = time.time()

    model = efficientnetv2_small("swish")
    optimizer = tfa.optimizers.AdamW(weight_decay=1e-7)
    #loss2 = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.15)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    schedule = WarmupCosineDecay(total_steps=16000, warmup_steps=1050, hold=550)

    history = model.fit(trainloader.repeat(), epochs=400, steps_per_epoch=40, validation_data=testloader,
                        callbacks=[TerminateOnNaN(), schedule],
                        verbose=0)

    f = np.nanmax(history.history['val_accuracy'])
    v = np.nanmax(history.history['val_loss'])
    finish = time.time()
    total = finish - start
    return f, v, total, history.history


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    logging.basicConfig(filename=args.logs_file, level=logging.DEBUG)

    import sys

    #for i in range(0, 3):
    #    f, v, t, hist = fitness_function_effnet2(None)
    #    print(f)
    #    logging.info(f)
    #sys.exit()
    algo = pickle.load(open(args.save_dir + "/" + args.save_file + "_effnet5", "rb"))
    #algo = pickle.load(open(args.save_file + "_effnet5", "rb"))

    fits = []
    for ind in algo.prev_individuals:
        if ind.hist is not None:
            fits.append(np.max(ind.hist['val_accuracy']))
        else:
            fits.append(-1)
    fits = np.asarray(fits)

    un_ind = {}
    for ind in np.asarray(algo.prev_individuals)[np.argsort(-fits)[0:200]]:
        if len(un_ind) == 100:
            break
        if ind.msg in un_ind:
            continue  # print(un_ind[ind.msg], np.max(ind.hist['val_accuracy']))
        else:
            un_ind[ind.msg] = ind  # np.max(ind.hist['val_accuracy'])

    res = {}
    if args.batch == 0:
        inds = np.asarray(list(un_ind.values())[0:50]) #np.asarray(algo.prev_individuals)[np.argsort(-fits)[0:50]]
        for i in range(0, len(inds)):
            print(i, inds[i].msg)
            logging.info(i)
            logging.info(inds[i].msg)
            r = []
            for j in range(0, 3):
                f, v, t, hist = fitness_function_effnet2(inds[i].call)
                print(f)
                logging.info(f)
                r.append(f)
            res[inds[i]] = r
        pickle.dump(res, open(args.save_dir + "/" + args.save_file + "_proj_res_0_49", "wb"))
        print("Done")
        logging.info("Done")
    else:
        inds = np.asarray(list(un_ind.values())[50:]) #np.asarray(algo.prev_individuals)[np.argsort(-fits)[50:100]]
        for i in range(0, len(inds)):
            print(i, inds[i].msg)
            logging.info(i)
            logging.info(inds[i].msg)
            r = []
            for j in range(0, 3):
                f, v, t, hist = fitness_function_effnet2(inds[i].call)
                print(f)
                logging.info(f)
                r.append(f)
            res[inds[i]] = r
        pickle.dump(res, open(args.save_dir + "/" + args.save_file + "_proj_res_50_100", "wb"))
        print("Done")
        logging.info("Done")
