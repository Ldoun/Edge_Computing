import argparse

def args_for_data(parser):
    parser.add_argument('--path', type=str, default='../data')
    parser.add_argument('--result_path', type=str, default='./result')
    
def args_for_train(parser):
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--epochs', type=int, default=200, help='max epochs')
    parser.add_argument('--patience', type=int, default=-1, help='patience for early stopping')    
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for the optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--warmup_epochs', type=int, default=-1, help='number of warmup epoch of lr scheduler')

def args_for_quantization(parser):
    parser.add_argument('--dense_model', type=str)
    parser.add_argument('--train_dense', action='store_true')
    parser.add_argument('--is_qat', action='store_true')
    parser.add_argument('--ch_quantize', action='store_true')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)

    args_for_data(parser)
    args_for_train(parser)
    args_for_quantization(parser)
    _args, _ = parser.parse_known_args()

    args = parser.parse_args()
    return args
