import argparse

from dense_net import DenseNet

train_params = {
    'batch_size': 8,
    'n_epochs': 80,
    'initial_learning_rate': 0.01,
    'reduce_lr_epoch_1': 60,  # epochs * 0.5
    'reduce_lr_epoch_2': 80,  # epochs * 0.75
    'reduce_lr_epoch_3': 100,
    'reduce_lr_epoch_4': 100,
    'validation_set': True,
    'validation_split': None,  # None or float 
    'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
    'normalization': None,  # None, divide_256, divide_255, by_chanels
}


def get_train_params():
        return train_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train', action='store_true',
        help='Train the model')
    parser.add_argument(
        '--test', action='store_true',
        help='Test model for required dataset if pretrained model exists.'
             'If provided together with `--train` flag testing will be'
             'performed right after training.')
    parser.add_argument(
        '--model_type', '-m', type=str, choices=['DenseNet', 'DenseNet-BC'],
        default='DenseNet-BC',
        help='What type of model to use')
    parser.add_argument(
        '--growth_rate', '-k', type=int, choices=[12, 24, 32, 40],
        default=24,
        help='Grows rate for every layer, '
             'choices were restricted to used in paper')
    parser.add_argument(
        '--depth', '-d', type=int, choices=[45, 63, 85, 121, 169, 201, 264],
        default=45,
        help='Depth of whole network, restricted to paper choices')
    parser.add_argument(
        '--total_blocks', '-tb', type=int, default=4, metavar='',
        help='Total blocks of layers stack (default: %(default)s)')
    parser.add_argument(
        '--keep_prob', '-kp', type=float, metavar='',
        help="Keep probability for dropout.")
    parser.add_argument(
        '--weight_decay', '-wd', type=float, default=1e-4, metavar='',
        help='Weight decay for optimizer (default: %(default)s)')
    parser.add_argument(
        '--nesterov_momentum', '-nm', type=float, default=0.9, metavar='',
        help='Nesterov momentum (default: %(default)s)')
    parser.add_argument(
        '--reduction', '-red', type=float, default=0.5, metavar='',
        help='reduction Theta at transition layer for DenseNets-BC models')

    parser.add_argument(
        '--logs', dest='should_save_logs', action='store_true',
        help='Write tensorflow logs')
    parser.add_argument(
        '--no-logs', dest='should_save_logs', action='store_false',
        help='Do not write tensorflow logs')
    parser.set_defaults(should_save_logs=True)

    parser.add_argument(
        '--saves', dest='should_save_model', action='store_true',
        help='Save model during training')
    parser.add_argument(
        '--no-saves', dest='should_save_model', action='store_false',
        help='Do not save model during training')
    parser.set_defaults(should_save_model=True)

    parser.add_argument(
        '--renew-logs', dest='renew_logs', action='store_true',
        help='Erase previous logs for model if exists.')
    parser.add_argument(
        '--not-renew-logs', dest='renew_logs', action='store_false',
        help='Do not erase previous logs for model if exists.')
    parser.set_defaults(renew_logs=True)

    args = parser.parse_args()

    if not args.keep_prob:
        args.keep_prob = 0.8
    if args.model_type == 'DenseNet':
        args.bc_mode = False
        args.reduction = 1.0
    elif args.model_type == 'DenseNet-BC':
        args.bc_mode = True

    model_params = vars(args)

    if not args.train and not args.test:
        print("You should train or test your network. Please check params.")
        exit()

    # some default params dataset/architecture related
    train_params = get_train_params()
    print("Params:")
    for k, v in model_params.items():
        print("\t%s: %s" % (k, v))
    print("Train params:")
    for k, v in train_params.items():
        print("\t%s: %s" % (k, v))

    print("Initialize the model..")
    model = DenseNet(**model_params)
    if args.train:
        # model.load_model()
        w1, w2 = model.train_all_epochs(train_params)
    if args.test:
        w1 = 0.5
        w2 = 0.5
        if not args.train:
            model.load_model()

        print("Testing...")
        _, _, _,_,_, _, _, _,_,_, _, acc1, acc2, accuracy = model.test(data_provider.test, w1, w2, batch_size=8)
        print("mean accuracy1: %f" % (acc1))
        print("mean accuracy2: %f" % (acc2))
        print("mean accuracy: %f" % (accuracy))
