import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import utils
import fire


# def tb_cleanup(logdir, backdir, threshold=100)
fire.Fire(utils.tb_cleanup)
