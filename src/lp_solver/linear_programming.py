import numpy as np
from scipy.optimize import linprog
import time

def lp_decode(received_message, channel_error_prob=0.1, r=5):
    return 0, 0