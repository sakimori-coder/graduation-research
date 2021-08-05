import numpy as np
from sporco.metric import psnr
from skimage.metrics import structural_similarity
import Option
import Convert
import util
from Solve import solve
from Prox import prox_l1

class Coef_update_L1():
    def __init__(self, opt, S):
        self.opt = opt
        self.S = S
        self.G0 = -self.S
        self.G1 = np.zeros((self.opt.K, self.opt.M, self.opt.N))
        self.H0 = np.zeros((self.opt.K, self.opt.N))
        self.H1 = np.zeros((self.opt.K, self.opt.M, self.opt.N))
        self.X = np.zeros((self.opt.K, self.opt.M, self.opt.N))

    def coef_update_l1(self, D):

        self.D = D
        self.Df = Convert.FFT_D(self.D ,self.opt)
        
        for i in range(self.opt.coef_iteration):
            self.H0_old = self.H0.copy()
            self.H1_old = self.H1.copy()
            self.X_update()
            self.G0_update()
            self.G1_update()
            self.H0_update()
            self.H1_update()

        return self.G1

    def X_update(self):
        for i in range(self.opt.K):
            G0SH0 = self.G0[i] + self.S[i] - self.H0[i]
            DtXf = util.create_Dtxf(self.Df, G0SH0, self.opt)
            G1H1 = self.G1[i] - self.H1[i]
            bf = DtXf + Convert.FFT_NMvector(G1H1, self.opt)
            self.X[i] = solve(self.Df, bf, 1, self.opt).real

    def G0_update(self):
        self.Dx = util.create_Dx(self.Df, self.X, self.opt)
        a = self.Dx -self.S + self.H0
        self.G0 = prox_l1(a, 1 / self.opt.Rho)

    def G1_update(self):
        a = self.X + self.H1
        self.G1 = prox_l1(a, self.opt.Lambda / self.opt.Rho)

    def H0_update(self):
        self.H0 = self.H0 + self.Dx - self.G0 - self.S

    def H1_update(self):
        self.H1 = self.H1 + self.X - self.G1



    def error_term(self):
        Dx = util.create_Dx(self.Df, self.G1, self.opt)
        return np.linalg.norm((Dx - self.S).flatten(), ord=1)

    def regularization_term(self):
        return self.opt.Lambda * np.linalg.norm(self.G1.flatten(), ord=1)

    def evaluation_function(self):
        return self.error_term() + self.regularization_term()

    def get_X(self):
        return self.G1

    def delta_dual_variable(self):
        return np.linalg.norm(self.H0-self.H0_old, ord=2) + np.linalg.norm(self.H1-self.H1_old, ord=2)

    def get_reconstruct(self):
        return util.create_Dx(self.Df, self.G1, self.opt).reshape(self.opt.K, int(np.sqrt(self.opt.N)), -1)

    def get_psnr_and_ssim(self):
        img = self.S.reshape(self.opt.K, int(np.sqrt(self.opt.N)), -1)
        reconst = self.get_reconstruct()
        for i in range(self.opt.K):
            print(i+1, "枚目")
            print("PSNR", psnr(img[i], reconst[i]))
            print("SSIM", structural_similarity(img[i], reconst[i], data_range=img[i].max()-img[i].min()))

    def get_L0(self):
        for i in range(self.opt.K):
            print(i+1, "枚目")
            print("L0 norm", np.sum(np.abs(self.G1[i]) != 0))

