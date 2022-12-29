from loss_projection import *


def preprocess_image(image, label):
    IMG_SHAPE = 32
    image = tf.image.resize(image, (IMG_SHAPE, IMG_SHAPE))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = (tf.cast(image, tf.float32) - 127.5) / 128

    return image, label


def basemodel_test(loss, use_imagenet=False, dataset=0):

    if dataset == 0:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

    AUTO = tf.data.experimental.AUTOTUNE
    BATCH_SIZE = 64  # 512
    trainloader_base = tf.data.Dataset.from_tensor_slices(
        (x_train, tf.keras.utils.to_categorical(y_train)))  # [train_ind_full]
    trainloader_base = (
        trainloader_base
        .map(preprocess_image, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    testloader = tf.data.Dataset.from_tensor_slices(
        (x_test, tf.keras.utils.to_categorical(y_test)))
    testloader = (
        testloader
        .map(preprocess_image, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    if use_imagenet:
        model = EfficientNetV2S(input_shape=(None, None, 3), num_classes=10,
                                classifier_activation='softmax', dropout=0.1)
        scheduler = WarmupCosineDecay(total_steps=16000, warmup_steps=1600, hold=0, target_lr=1e-3)
        epochs = 160
    else:
        model = EfficientNetV2S(input_shape=(None, None, 3), num_classes=10, pretrained=None,
                                classifier_activation='softmax', dropout=0.1)
        scheduler = WarmupCosineDecay(total_steps=64000, warmup_steps=6400, hold=0, target_lr=1e-3)
        epochs = 4 * 160

    optimizer = tf.optimizers.Adam()
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    history = model.fit(
        trainloader_base.repeat(),
        epochs=epochs,  # 160
        steps_per_epoch=100,
        validation_data=testloader,
        callbacks=[scheduler],
        verbose=1
    )
    return history.history



def create_parser():
    '''
    Create a command line parser for the XOR experiment
    '''
    parser = argparse.ArgumentParser(description='Neural Loss Evolution')
    parser.add_argument('--logs_file', type=str, default='cat_test_base_cut_prog.log',
                        help='Output File For Logging')
    parser.add_argument('--batch', type=int, default=0,
                        help='Save Directory for saving Logs/Checkups')
    parser.add_argument("--dataset", type=int, default=0,
                        help="dataset")
    return parser


def baikal_loss(y, yhat):
    eps = 1e-7
    return -tf.math.reduce_mean(tf.math.reduce_sum(tf.math.log(yhat+eps) - tf.math.divide(y, yhat+eps), axis=-1))


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    logging.basicConfig(filename=args.logs_file, level=logging.DEBUG)

    if args.batch == -2:
        loss = tf.keras.losses.CategoricalCrossentropy()
        base = "categorical_cross"
    elif args.batch == -1:
        loss = baikal_loss
        base = "baikal"
    else:
        loss = pickle.load("final_4_losses", "rb")[args.batch].call
        base = "best_loss_{}".format(args.batch)
        
    msg = "--- Base Model ---"
    print(msg)
    logging.info(msg)
    histories = []
    for i in range(0, 6):
        hist = basemodel_test(loss, use_imagenet=False, dataset=args.dataset)
        msg = " Iter: {} - Acc: {}".format(i+1, np.max(hist['val_accuracy']))
        print(msg)
        logging.info(msg)
        histories.append(hist)
    pickle.dump(histories, open("{}_base_test".format(base), "wb"))

    msg = "--- Base Model ImageNet ---"
    print(msg)
    logging.info(msg)
    histories = []
    for i in range(0, 6):
        hist = basemodel_test(loss, use_imagenet=True, dataset=args.dataset)
        msg = " Iter: {} - Acc: {}".format(i + 1, np.max(hist['val_accuracy']))
        print(msg)
        logging.info(msg)
        histories.append(hist)
    pickle.dump(histories, open("{}_base_test_imagenet".format(base), "wb"))

    msg = "--- Progressive Model ---"
    print(msg)
    logging.info(msg)
    histories = []
    for i in range(0, 6):
        hist = progressive_test(loss, use_imagenet=False, test=True, dataset=args.dataset)
        msg = " Iter: {} - Acc: {}".format(i + 1, np.max(hist['val_accuracy']))
        print(msg)
        logging.info(msg)
        histories.append(hist)
    pickle.dump(histories, open("{}_progressive_test".format(base), "wb"))

    msg = "--- Progressive Model ImageNet---"
    print(msg)
    logging.info(msg)
    histories = []
    for i in range(0, 6):
        hist = progressive_test(loss, use_imagenet=True, test=True, dataset=args.dataset)
        msg = " Iter: {} - Acc: {}".format(i + 1, np.max(hist['val_accuracy']))
        print(msg)
        logging.info(msg)
        histories.append(hist)
    pickle.dump(histories, open("{}_progressive_test_imagenet".format(base), "wb"))






