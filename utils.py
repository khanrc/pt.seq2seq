import os
from datetime import datetime


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class BestTracker():
    def __init__(self, track_type, value=None):
        assert track_type in ['min', 'max']
        self.type = track_type
        limit_value = 99999999. if track_type == 'min' else -99999999.
        self.val = value or limit_value
        self.ep = None

    def better(self, new):
        if self.type == 'min':
            return new < self.val
        else:
            return new > self.val

    def check(self, value, epoch=None):
        if self.better(value):
            self.val = value
            self.ep = epoch


def makedirs(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def timestamp(fmt="%y%m%d_%H-%M-%S"):
    return datetime.now().strftime(fmt)


###############################
###### Parameter tracing ######
###############################

def num_params(module):
    return sum(p.numel() for p in module.parameters())

#  # sequential tracing
#  def seq_param_trace(module):
#      for name, p in module.named_parameters():
#          numel = p.numel()
#          unit = 'M'
#          numel /= 1024*1024
#          fmt = "10.3f" if numel < 1.0 else "10.1f"

#          print("{:50s}\t{:{fmt}}{}".format(name, numel, unit, fmt=fmt))

#  # recursive tracing
#  def param_trace(name, module, depth, max_depth=999, threshold=0):
#      if depth > max_depth:
#          return
#      prefix = "  " * depth
#      n_params = num_params(module)
#      if n_params > threshold:
#          print("{:60s}\t{:10.3f}M".format(prefix + name, n_params / 1024 / 1024))
#      for n, m in module.named_children():
#          if depth == 0:
#              child_name = n
#          else:
#              child_name = "{}.{}".format(name, n)
#          param_trace(child_name, m, depth+1, max_depth, threshold)
