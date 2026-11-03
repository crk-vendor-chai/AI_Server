"""Small RGB appearance prototype classifier shared by fit, calibration and runtime."""
import math

import numpy as np


def transform(centers,gamma,row):
    delta=np.asarray(centers,dtype=np.float32)-np.asarray(row,dtype=np.float32)
    distance=np.sum(delta*delta,axis=1,dtype=np.float32)
    return np.exp(-np.float64(gamma)*distance.astype(np.float64))


def score(centers,model,row):
    f=transform(centers,model['gamma'],row)
    logit=float(f@np.asarray(model['coef'],dtype=np.float64))+model['intercept']
    return 1./(1.+math.exp(-max(-700.,min(700.,logit))))
