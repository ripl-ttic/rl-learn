from rl_learn.util import load_gin_configs
from rl_learn.algorithms import RunRLLEARN
import argparse, os

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train RL LEARN.')
    parser.add_argument('--expt_id', type=int, help='task id')
    parser.add_argument('--descr_id', type=int, help='description id')
    parser.add_argument('--lang_coef', type=float, help='lang coef')
    parser.add_argument('--lang_enc', type=str, default='onehot', help='lang enc')
    parser.add_argument('-c', '--gin_config', type=str, help='gin config')
    parser.add_argument('-b', '--gin_bindings', nargs='+', help='gin bindings to overwrite config')
    args = parser.parse_args()
    if args.gin_config is None:
        config = os.path.dirname(os.path.dirname(__file__)) + 'configs/rllearn.gin'
    else:
        config = args.gin_config
    load_gin_configs([config], args.gin_bindings)

    learn = RunRLLEARN(args.lang_enc, args.expt_id, args.descr_id, args.lang_coef)
    learn.train()