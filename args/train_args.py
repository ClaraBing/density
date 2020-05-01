from .base_args import BaseArgs

def str2bool(s):
    return s.lower().startswith('t')

class TrainArgs(BaseArgs):
  def __init__(self):
    super(TrainArgs, self).__init__()
    self.is_train = True

    # Training
    self.parser.add_argument('--benchmark', type=str2bool, default=True, help='Turn on CUDNN benchmarking')
    self.parser.add_argument('--optim', type=str, default='SGD')
    self.parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate.')
    self.parser.add_argument('--wd', default=1e-5, type=float, help="Weight decay.")
    self.parser.add_argument('--use-val', type=int, help="Whether to use a val set during training.")
    self.parser.add_argument('--max_grad_norm', type=float, default=-1., help='Max gradient norm for clipping')
    self.parser.add_argument('--num-epochs', default=100, type=int, help='Number of epochs to train')
    self.parser.add_argument('--num-samples', default=64, type=int, help='Number of samples at test time')
   

