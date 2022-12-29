from effnet_rand_aug import *

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

train_ind = list(range(0, 45000))
val_ind = list(range(45000, 50000))

AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 128  # 512
IMG_SHAPE = 32


def preprocess_image(image, label):
    image = tf.image.resize(image, (IMG_SHAPE, IMG_SHAPE))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image / 255.
    return image, label


trainloader = tf.data.Dataset.from_tensor_slices(
    (x_train[train_ind], tf.keras.utils.to_categorical(y_train[train_ind])))
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


# for resnet

train_process = RandomProcessImage((80, 80), magnitude=5, keep_shape=True)  
test_process = RandomProcessImage((80, 80), magnitude=-1, keep_shape=True)

trainloader_randaug_80 = tf.data.Dataset.from_tensor_slices(
    (x_train[train_ind], tf.keras.utils.to_categorical(y_train[train_ind]))).shuffle(
    1024).map(train_process,
              num_parallel_calls=AUTO).batch(64).prefetch(buffer_size=AUTO)
testloader_randaug_80 = tf.data.Dataset.from_tensor_slices(
    (x_train[val_ind], tf.keras.utils.to_categorical(y_train[val_ind]))).shuffle(
    1024).map(test_process,
              num_parallel_calls=AUTO).batch(64).prefetch(buffer_size=AUTO)

# for effnet

train_process = RandomProcessImage((96, 96), magnitude=5, keep_shape=True)  
test_process = RandomProcessImage((96, 96), magnitude=-1, keep_shape=True)

trainloader_randaug_96 = tf.data.Dataset.from_tensor_slices(
    (x_train[train_ind], tf.keras.utils.to_categorical(y_train[train_ind]))).shuffle(
    1024).map(train_process,
              num_parallel_calls=AUTO).batch(64).prefetch(buffer_size=AUTO)
testloader_randaug_96 = tf.data.Dataset.from_tensor_slices(
    (x_train[val_ind], tf.keras.utils.to_categorical(y_train[val_ind]))).shuffle(
    1024).map(test_process,
              num_parallel_calls=AUTO).batch(64).prefetch(buffer_size=AUTO)


def ConvNet(act, shape=(32, 32, 3)):
    inputs = tf.keras.layers.Input(shape=shape)
    x = tf.keras.layers.Conv2D(12, (3, 3), padding='same')(inputs)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(act)(x)
    x = tf.keras.layers.Conv2D(24, (3, 3), strides=(2, 2), padding='same')(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(act)(x)
    x = tf.keras.layers.Conv2D(48, (3, 3), strides=(2, 2), padding='same')(x)

    x = tf.keras.layers.GlobalAvgPool2D()(x)
    x = tf.keras.layers.Dropout(0.10)(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model


def block(prev, filter, act):
    x = tf.keras.layers.BatchNormalization()(prev)
    x = tf.keras.layers.Activation(act)(x)
    x = tf.keras.layers.Conv2D(filter, (3, 3), strides=(2, 2), padding='same')(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(act)(x)
    x = tf.keras.layers.Conv2D(filter, (3, 3), padding='same')(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(act)(x)
    x = tf.keras.layers.Conv2D(filter, (3, 3), padding='same')(x)

    prev = tf.keras.layers.Conv2D(filter, (1, 1), padding='same', strides=(2, 2))(prev)

    x = tf.keras.layers.Add()([x, prev])

    return x


def build_resnet9_v2(act, shape=(32, 32, 3)):
    inputs = tf.keras.layers.Input(shape=shape)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same')(inputs)

    x = block(x, 16, act)
    x = block(x, 32, act)

    x = tf.keras.layers.GlobalAvgPool2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model


class EpochCallback(tf.keras.callbacks.Callback):

    def __init__(self, threshold, epoch=6):
        super(EpochCallback, self).__init__()
        self.threshold = threshold
        self.epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.epoch:
            if logs['accuracy'] < self.threshold:  # 30
                self.model.stop_training = True


def fitness_function_resnet_standard(loss):
    start = time.time()
    threshold = 0.35
    model = build_resnet9_v2("swish")
    optimizer = tf.optimizers.Adam()
    scheduler = WarmupCosineDecay(total_steps=8000, warmup_steps=800, hold=0, target_lr=1e-3)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    history = model.fit(trainloader.repeat(), epochs=80, steps_per_epoch=100, validation_data=testloader,
                        callbacks=[TerminateOnNaN(), scheduler, EpochCallback(threshold=threshold)],
                        verbose=0)

    f = np.nanmax(history.history['val_accuracy'])
    v = np.nanmax(history.history['val_loss'])
    if np.nanmax(history.history['accuracy']) < threshold:
        return None
    finish = time.time()
    total = finish - start
    return f, v, total, history.history


def fitness_function_resnet_rand_aug(loss):
    start = time.time()
    threshold = 0.35
    model = build_resnet9_v2("swish")
    optimizer = tf.optimizers.Adam()
    scheduler = WarmupCosineDecay(total_steps=8000, warmup_steps=800, hold=0, target_lr=1e-3)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    history = model.fit(trainloader.repeat(), epochs=8, steps_per_epoch=100, validation_data=testloader,
                        callbacks=[TerminateOnNaN(), scheduler],
                        verbose=0)

    if np.nanmax(history.history['accuracy']) < threshold:
        return None

    threshold = 0.22
    model = build_resnet9_v2("swish", shape=(None, None, 3))
    optimizer = tf.optimizers.Adam()
    scheduler = WarmupCosineDecay(total_steps=8000, warmup_steps=800, hold=0, target_lr=1e-3)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    history = model.fit(trainloader_randaug_80.repeat(), epochs=80, steps_per_epoch=100,
                        validation_data=testloader_randaug_80,
                        callbacks=[TerminateOnNaN(), scheduler, EpochCallback(threshold=threshold, epoch=7)],
                        verbose=0)

    f = np.nanmax(history.history['val_accuracy'])
    v = np.nanmax(history.history['val_loss'])
    finish = time.time()
    total = finish - start
    return f, v, total, history.history


def fitness_function_convnet_standard(loss):
    start = time.time()
    threshold = 0.37
    model = ConvNet("swish")
    optimizer = tf.optimizers.Adam()
    scheduler = WarmupCosineDecay(total_steps=8000, warmup_steps=800, hold=0, target_lr=1e-3)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    history = model.fit(trainloader.repeat(), epochs=80, steps_per_epoch=100, validation_data=testloader,
                        callbacks=[TerminateOnNaN(), scheduler, EpochCallback(threshold=threshold, epoch=8)],
                        verbose=0)

    f = np.nanmax(history.history['val_accuracy'])
    v = np.nanmax(history.history['val_loss'])
    if np.nanmax(history.history['accuracy']) < threshold:
        return None
    finish = time.time()
    total = finish - start
    return f, v, total, history.history


def fitness_function_effnet_rand_aug(loss):
    threshold = 0.37
    model = ConvNet("swish")
    optimizer = tf.optimizers.Adam()
    scheduler = WarmupCosineDecay(total_steps=8000, warmup_steps=800, hold=0, target_lr=1e-3)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    history = model.fit(trainloader.repeat(), epochs=8, steps_per_epoch=100, validation_data=testloader,
                        callbacks=[TerminateOnNaN(), scheduler],
                        verbose=0)

    if np.nanmax(history.history['accuracy']) < threshold:
        return None

    start = time.time()
    threshold = 0.25

    model = EfficientNetV2S(input_shape=(None, None, 3), num_classes=10, pretrained=None,
                            classifier_activation='softmax', dropout=0.1)
    optimizer = tf.optimizers.Adam()
    scheduler = WarmupCosineDecay(total_steps=64000, warmup_steps=6400, hold=0, target_lr=1e-3)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    history = model.fit(trainloader_randaug_96.repeat(), epochs=80, steps_per_epoch=100,
                        validation_data=testloader_randaug_96,
                        callbacks=[TerminateOnNaN(), scheduler, EpochCallback(threshold=threshold, epoch=30)],
                        verbose=0)

    f = np.nanmax(history.history['val_accuracy'])
    v = np.nanmax(history.history['val_loss'])
    finish = time.time()
    total = finish - start
    return f, v, total, history.history