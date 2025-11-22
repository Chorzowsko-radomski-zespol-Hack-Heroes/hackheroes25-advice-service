from neural_net import recommendations
import numpy as np
def get_jobs(inp, wpep_mode, j_print): #inp=[0.12, 0.98, 0.44...]
    pers_input=np.array(inp)
    top=recommendations(pers_input, wpep_mode, j_print)
    return top #top[0][i]=jobs, top[1][i]=scores for jobs(needs to be *100 to return precent)
out=get_jobs([0.66,0.64,0.01,0.28,0.52,0.64,0.93,0.6,0.95,0.41,0.7,0.89,0.01,0.68,0.42,0.84,0.45,0.99,0.61,0.55,0.5,0.52,0.62,0.84,0.82], 1, 5)
for i in range(len(out[0])):
    print(out[0][i], out[1][i])