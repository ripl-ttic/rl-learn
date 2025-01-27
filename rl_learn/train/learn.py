from rl_learn.util import load_gin_configs
from rl_learn.algorithms import RunLEARN
import argparse, os

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train LEARN.')
    parser.add_argument('--lang_enc', type=str, default='onehot', help='lang enc')
    parser.add_argument('-c', '--gin_config', type=str, help='gin config')
    parser.add_argument('-b', '--gin_bindings', nargs='+', help='gin bindings to overwrite config')
    args = parser.parse_args()
    if args.gin_config is None:
        config = os.path.dirname(os.path.dirname(__file__)) + 'configs/learn.gin'
    else:
        config = args.gin_config
    load_gin_configs([config], args.gin_bindings)

    learn = RunLEARN(args.lang_enc)
    learn.train()