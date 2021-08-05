import numpy as np
from sporco.metric import psnr
from skimage.metrics import structural_similarity
from sporco.util import tiledict
from sporco.plot import imview
import Option
import util
import Convert
from Solve import solve
import Prox

class Dict_update_L1():
    def __init__(self, opt, S):
        self.opt = opt
        self.S = S
        self.G0 = -self.S
        self.G1 = np.zeros((self.opt.M, self.opt.N))
        self.H0 = np.zeros((self.opt.K, self.opt.N))
        self.H1 = np.zeros((self.opt.M, self.opt.N))

    def dict_update_l1(self, X):

        self.Xf = Convert.FFT_X(X, self.opt)

        for i in range(self.opt.dict_iteration):
            self.H0_old = self.H0.copy()
            self.H1_old = self.H1.copy()
            self.D_update()
            self.G0_update()
            self.G1_update()
            self.H0_update()
            self.H1_update()

        return self.G1

    def D_update(self):
        Xf_sum = np.sum(self.Xf, axis=0)
        S_sum = np.sum(self.S, axis=0)
        G0_sum = np.sum(self.G0, axis=0)
        H0_sum = np.sum(self.H0, axis=0)
        G0FH0 = G0_sum + S_sum - H0_sum
        Xtdf = util.create_Dtxf(Xf_sum, G0FH0, self.opt)
        G1H1 = self.G1 - self.H1
        bf = Xtdf + Convert.FFT_NMvector(G1H1, self.opt)
        self.D = solve(Xf_sum, bf, 1, self.opt).real

    def G0_update(self):
        self.Xd = util.create_Xd(self.Xf, self.D, self.opt)
        a = self.Xd -self.S + self.H0
        self.G0 = Prox.prox_l1(a, 1 / self.opt.Rho)

    def G1_update(self):
        a = self.D + self.H1
        PtPy = Prox.prox_projection(a, self.opt)
        for i in range(self.opt.M):
            norm = np.linalg.norm(PtPy[i], ord=2)
            if(norm > 1):
                PtPy[i] = PtPy[i] / norm
        self.G1 = PtPy

    def H0_update(self):
        self.H0 = self.H0 + self.Xd - self.G0 - self.S

    def H1_update(self):
        self.H1 = self.H1 + self.D - self.G1




    def error_term(self):
        Xd = util.create_Dx(self.Xf, self.G1, self.opt)
        return np.linalg.norm((Xd - self.S).flatten(), ord=1)

    def get_D(self):
        return self.D

    def get_tiled_D(self):
        D_copy = self.D.reshape(self.opt.M, int(np.sqrt(self.opt.N)), -1)
        D_copy = D_copy[:,0:self.opt.Filter_Width, 0:self.opt.Filter_Width]
        D_copy = np.transpose(D_copy,(1, 2, 0))
        imview(tiledict(D_copy), fgsz=(7,7), title='Dictionary')

    def delta_dual_variable(self):
        return np.linalg.norm(self.H0-self.H0_old, ord=2) + np.linalg.norm(self.H1-self.H1_old, ord=2)

    def get_reconstruct(self):
        return util.create_Dx(self.Xf, self.G1, self.opt).reshpae(self.opt.K, int(np.sqrt(self.opt.N)), -1)

    def get_psnr_and_ssim(self):
        img = self.S.reshape(self.opt.K, int(np.sqrt(self.opt.N)), -1)
        reconst = self.get_reconstruct()
        for i in range(self.opt.K):
            print(i+1, "枚目")
            print("PSNR", psnr(self.img[i], reconst[i]))
            print("SSIM", structural_similarity(img[i], reconst[i], data_range=img[i].max() - img[i].min()))
            


