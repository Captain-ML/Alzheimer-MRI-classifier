import numpy as np
class Argmax:
    def __init__(self, arr):
        self.pred_arr = arr
        self.pred_list = []  # Instance variable (not shared across instances)

    def get_arg(self):
        for row in self.pred_arr:
            self.pred_list.append(np.argmax(row))

    def print_list(self):
        return self.pred_list