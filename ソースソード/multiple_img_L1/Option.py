import numpy as np
class DictLearn_Option():
    def __init__(self, N, M, K, Filter_Width, Lambda, Rho, coef_iteration, dict_iteration):
        self.N = N
        self.M = M
        self.K = K
        self.Filter_Width = Filter_Width
        self.Lambda = Lambda
        self.Rho = Rho
        self.coef_iteration = coef_iteration
        self.dict_iteration = dict_iteration