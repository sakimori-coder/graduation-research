import numpy as np
import Option

def prox_l1(x, alpha):
    '''
    L1のprox(ソフトしきい値関数)
    '''
    return np.sign(x) * (np.clip(np.abs(x) - alpha, 0, float('Inf')))

def prox_projection(x, opt):
    '''
    CPNの指示関数のprox(距離射影)
    '''
    PtP = np.zeros((int(np.sqrt(opt.N)),int(np.sqrt(opt.N))))
    PtP[0:opt.Filter_Width,0:opt.Filter_Width] = 1
    return x * PtP.reshape(opt.N)