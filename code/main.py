from code.career_adviser import get_jobs
from neural_net import recommendations
import numpy as np

if __name__ == "__main__":
    pers_input = [0.35, 0.8, 0.8, 0.9, 0.68, 0.6, 0.56, 0.48,  # psychology
                  0.03, 0.4, 0.96, 0.6, 0.93, 0.5,  # ostatnie: medycyna/biologia
                  0.76, 0.77, 0.7, 0.45, 0.36, 0.5,  # ostatnie: wystąpienia publiczne
                  0.7, 0.6, 0.90, 0.3, 0.8  # ostatnie: praca z ludźmi
                  ]

    out = get_jobs(pers_input, 1, 5)
    for i in range(len(out[0])):
        print(out[0][i], out[1][i] * 100)
