from neural_net import recommendations
import numpy as np


def get_jobs(inp, wpep_mode):  # inp=[0.12, 0.98, 0.44...]
    pers_input = np.array(inp)
    top = recommendations(pers_input, wpep_mode, 5)
    # top[0][i]=jobs, top[1][i]=scores for jobs(needs to be *100 to return precent)
    return top


def pretty_print_jobs(jobs, scores):
    # scores zakładam 0–1, ale mogą być bardzo małe — normalizujemy do %.
    scores = np.array(scores, dtype=float)
    if scores.max() > 0:
        norm = scores / scores.max()
    else:
        norm = scores

    print("\n=== TOP REKOMENDACJE ===")
    for j, s, n in zip(jobs, scores, norm):
        pct = round(n * 100, 1)
        raw = f"{s:.3e}"
        print(f"- {j:<30}  |  {pct:>5}%   (raw={raw})")
    print("=========================\n")


if __name__ == "__main__":
    # pers_input = np.array([0.35, 0.8, 0.8, 0.9, 0.68, 0.6, 0.56, 0.48,  # psychology
    #                        0.03, 0.4, 0.96, 0.6, 0.95, 0.1,  # ostatnie: medycyna/biologia
    #                        0.76, 0.77, 0.7, 0.45, 0.36, 0.5,  # ostatnie: wystąpienia publiczne
    #                        0.7, 0.6, 0.90, 0.3, 0.8  # ostatnie: praca z ludźmi
    #                        ])
    pers_input = np.array([0.9, 0.1, 0.3, 0.2, 0.8, 0.5, 0.3, 0.4,  # psychology
                           0.3, 0.7, 0.96, 0.6, 0.1, 0.1,  # ostatnie: medycyna/biologia
                           0.1, 0.77, 0.7, 0.45, 0.36, 0.9,  # ostatnie: wystąpienia publiczne
                           0.3, 0.3, 0.4, 0.5, 0.9  # ostatnie: praca z ludźmi
                           ])
    jobs, scores = get_jobs(pers_input, 0)
    pretty_print_jobs(jobs, scores)
