import argparse

def args_for_data(parser):
    parser.add_argument('--path', type=str, default='../data')
    parser.add_argument('--result_path', type=str, default='./result')
    
def args_for_train(parser):
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
    parser.add_argument('--batch_size', type=int, default=None, help='batch_size')
    parser.add_argument('--epochs', type=int, default=1000, help='max epochs')
    parser.add_argument('--patience', type=int, default=15, help='patience for early stopping')    
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate for the optimizer')
    parser.add_argument('--scheduler', type=str, default='None')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='number of warmup epoch of lr scheduler')

def args_for_pruning(parser):
    parser.add_argument('--prune_type', type=str, default='unstructured')
    parser.add_argument('--pruning_ratio', type=float, default=0.98)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)

    args_for_data(parser)
    args_for_train(parser)
    args_for_pruning(parser)
    _args, _ = parser.parse_known_args()

    args = parser.parse_args()
    return args
