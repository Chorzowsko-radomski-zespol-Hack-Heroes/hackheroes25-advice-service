from neural_net import recommendations
import numpy as np
def get_jobs(inp, wpep_mode, j_print): #inp=[0.12, 0.98, 0.44...]
    pers_input=np.array(inp)
    top=recommendations(pers_input, wpep_mode, j_print)
    return top #top[0][i]=jobs, top[1][i]=scores for jobs(needs to be *100 to return precent)