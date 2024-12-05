from model import Temp

def args_for_model(parser, model):
    parser.add_argument('--something', type=str, default="?", help="?")
