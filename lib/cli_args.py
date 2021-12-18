import argparse


def get_args(data_names: list, config: dict) -> tuple:
    parser = argparse.ArgumentParser(description="Model training script.")
    parser.add_argument(
        '-m', 
        help='Pick which model to train (default: vae).', 
        default='vae',
        type=str,
        choices=['lvae', 'vae', 'draw', 'flow'],
        dest='model'
    )
    if data_names:
        parser.add_argument(
            '-d', 
            help='Pick which dataset to fit to (default: 8gaussians).', 
            default='8gaussians',
            type=str,
            choices=data_names,
            dest='dataset'
        )
    parser.add_argument(
        '-e', 
        help='Pick number of epochs to train over (default: 100).', 
        default=100,
        type=int,
        dest='epochs'
    )
    parser.add_argument(
        '-mute', 
        help='Mute tqdm outputs. (mainly for bsub submit runs)', 
        action='store_true'
    )
    parser.add_argument(
        '-n',
        help='Pick number of models to train.', 
        default=1,
        type=int,
        dest='n_runs'
    )
    args = parser.parse_args()

    # add args to config
    config['model'] = args.model
    config['epochs'] = args.epochs
    if 'dataset' in args:
        config['dataset'] = args.dataset

    # leftover args
    args = {'n_runs': args.n_runs, 'mute': args.mute}

    return config, args
