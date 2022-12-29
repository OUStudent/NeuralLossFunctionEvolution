from fitness_functions import *


def progressive_test(loss, target_shapes=(128, 160, 192, 224),
                     dropouts=(0.1, 0.2, 0.3, 0.4),
                     magnitudes=(5, 8, 12, 15), use_imagenet=False, test=False, dataset=0):
    if use_imagenet:
        print("Using Imagenet")
        model = EfficientNetV2S(input_shape=(None, None, 3), num_classes=10,
                                classifier_activation='softmax', dropout=0.1)
    else:
        print("Random Init")
        model = EfficientNetV2S(input_shape=(None, None, 3), num_classes=10, pretrained=None,
                                classifier_activation='softmax', dropout=0.1)

    if dataset == 0:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

    optimizer = tf.optimizers.Adam()
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    stages = min([len(target_shapes), len(dropouts), len(magnitudes)])
    temp_hist = []
    scheduler = WarmupCosineDecay(total_steps=64000, warmup_steps=6400, hold=0, target_lr=1e-3)
    for stage, target_shape, dropout, magnitude in zip(range(stages), target_shapes, dropouts, magnitudes):
        print(">>>> stage: {}/{}, target_shape: {}, dropout: {}, magnitude: {}".format(stage + 1, stages, target_shape,
                                                                                       dropout, magnitude))
        if len(dropouts) > 1 and isinstance(model.layers[-2], keras.layers.Dropout):
            print(">>>> Changing dropout rate to:", dropout)
            model.layers[-2].rate = dropout

        train_process = RandomProcessImage((target_shape, target_shape), magnitude=magnitude, keep_shape=True)
        test_process = RandomProcessImage((target_shape, target_shape), magnitude=-1, keep_shape=True)
        if test:
            print("using Test Set")
            trainloader_randaug = tf.data.Dataset.from_tensor_slices(
                (x_train, tf.keras.utils.to_categorical(y_train))).shuffle(  # [train_ind]
                1024).map(train_process,
                          num_parallel_calls=AUTO).batch(64).prefetch(buffer_size=AUTO)
            testloader_randaug = tf.data.Dataset.from_tensor_slices(
                (x_test, tf.keras.utils.to_categorical(y_test))).shuffle(  # y_train[val_ind]
                1024).map(test_process,
                          num_parallel_calls=AUTO).batch(64).prefetch(buffer_size=AUTO)
        else:
            print("Using Validation set")
            trainloader_randaug = tf.data.Dataset.from_tensor_slices(
                (x_train[train_ind], tf.keras.utils.to_categorical(y_train[train_ind]))).shuffle(  # [train_ind]
                1024).map(train_process,
                          num_parallel_calls=AUTO).batch(64).prefetch(buffer_size=AUTO)
            testloader_randaug = tf.data.Dataset.from_tensor_slices(
                (x_train[val_ind], tf.keras.utils.to_categorical(y_train[val_ind]))).shuffle(  # y_train[val_ind]
                1024).map(test_process,
                          num_parallel_calls=AUTO).batch(64).prefetch(buffer_size=AUTO)

        initial_epoch = 0  # stage * 160
        # epochs = (stage + 1) * total_epochs // stages
        epochs = 160  # 160  # 60
        history = model.fit(
            trainloader_randaug.repeat(),
            epochs=epochs,  # 160
            steps_per_epoch=100,
            initial_epoch=initial_epoch,
            validation_data=testloader_randaug,
            callbacks=[scheduler, EpochCallback(threshold=0.45, epoch=40)],
            verbose=1
        )
        f = np.nanmax(history.history['val_accuracy'])
        print(" Val Acc: {}".format(f))
        temp_hist.append(history)
    hist = {kk: np.ravel([hh.history[kk] for hh in temp_hist]).astype("float").tolist() for kk in
            history.history.keys()}
    return hist


def create_parser():
    '''
    Create a command line parser for the XOR experiment
    '''
    parser = argparse.ArgumentParser(description='Neural Loss Evolution')
    parser.add_argument('--logs_file', type=str, default='cat_test_base_cut_prog.log',
                        help='Output File For Logging')
    parser.add_argument('--batch', type=int, default=0,
                        help='Save Directory for saving Logs/Checkups')

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    logging.basicConfig(filename=args.logs_file, level=logging.DEBUG)
    loss = tf.keras.losses.CategoricalCrossentropy()

    if args.batch == 0:
        inds = pickle.load(open("effnet_best_150", "rb"))[0:75]
    elif args.batch == 1:
        inds = pickle.load(open("effnet_best_150", "rb"))[75:]
    elif args.batch == 2:
        inds = pickle.load(open("resnet_best_300", "rb"))[150]
    else:
        inds = pickle.load(open("resnet_best_300", "rb"))[150:]

    target_shapes = [(128),
                     (128, 160),
                     (128, 160, 192),
                     (128, 160, 192, 224)]
    dropouts = [(0.1),
                (0.1, 0.2),
                (0.1, 0.2, 0.3),
                (0.1, 0.2, 0.3, 0.4)]
    magnitudes = [(5),
                  (5, 8),
                  (5, 8, 12),
                  (5, 8, 12, 15)]
    sizes = [25, 12, 6, 4]

    for i in range(0, 4):

        start = time.time()
        msg = "--- Starting Phase {} ---".format(i+1)
        logging.info(msg)
        print(msg)
        res = []
        for j, ind in enumerate(inds):
            msg = "{}, {}".format(j, ind.msg)
            print(msg)
            logging.info(msg)
            if i == 0:
                hist = progressive_test(ind.call, target_shapes=[target_shapes[i]],
                                        dropouts=[dropouts[i]],
                                        magnitudes=[magnitudes[i]])
            else:
                hist = progressive_test(ind.call, target_shapes=target_shapes[i],
                                        dropouts=dropouts[i],
                                        magnitudes=magnitudes[i])
            f = np.max(hist['val_accuracy'])
            print(f)
            logging.info(f)
            res.append(f)
        res = np.asarray(res)
        pickle.dump([inds, res], open("results_batch_{}_phase_{}".format(args.batch, i+1), "wb"))
        bst = np.argsort(-res)[0:sizes[i]]
        inds = inds[bst]
        end = time.time()
        msg = "Ending Phase {}... Time Taken: {}".format(i+1, end-start)




