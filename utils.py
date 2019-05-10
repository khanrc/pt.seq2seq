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


def makedirs(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def tb_name(comment):
    """ Generate tensorboard run name
    Just remove hostname from default.
    """
    cur_time = datetime.now().strftime('%b%d_%H-%M-%S')
    name = "{}_{}".format(cur_time, comment)
    log_dir = os.path.join('runs', name)
    return log_dir
