from code.neural_net_lite import recommendations_tflite
import numpy as np


def get_jobs(inp, wpep_mode, j_print):  # inp=[0.12, 0.98, 0.44...]
    pers_input = np.array(inp)
    top = recommendations_tflite(pers_input, wpep_mode, j_print)
    # top[0][i]=jobs, top[1][i]=scores for jobs(needs to be *100 to return precent)
    return top
