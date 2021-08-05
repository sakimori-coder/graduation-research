# %%
import numpy as np
from tqdm import tqdm
import Option
import util
import Convert
from Coef_update_L1 import Coef_update_L1
from Dict_update_L1 import Dict_update_L1
from sporco import plot
from sporco.util import ExampleImages
from PIL import Image

# %%
np.random.seed(seed=123)
dictlearn_opt = Option.DictLearn_Option(N=128*128, M=30, K=1, Filter_Width=8, Lambda=0.005, Rho=1, coef_iteration=2, dict_iteration=2)
D = np.random.normal(0, 1, (dictlearn_opt.M, dictlearn_opt.Filter_Width, dictlearn_opt.Filter_Width))
D = util.padding(D, dictlearn_opt)
X = np.zeros((dictlearn_opt.K, dictlearn_opt.M, dictlearn_opt.N))

S = np.zeros((dictlearn_opt.K, dictlearn_opt.N))
exim = ExampleImages(scaled=True, zoom=0.25, gray=True)
S[0] = exim.image('barbara.png', idxexp=np.s_[10:522, 100:612]).flatten()
#S[1] = exim.image('kodim23.png', idxexp=np.s_[:, 60:572]).flatten()
#S[2] = exim.image('monarch.png', idxexp=np.s_[:, 160:672]).flatten()

for i in range(dictlearn_opt.K):
    S[i] = util.normalize(S[i])

Coef = Coef_update_L1(dictlearn_opt, S)
Dict = Dict_update_L1(dictlearn_opt, S) 
Dict.D = D
Dict.get_tiled_D() 

# %%
for i in tqdm(range(500)):
    X = Coef.coef_update_l1(D)
    D = Dict.dict_update_l1(X)

Dict.get_tiled_D()
# %%
test_opt = Option.DictLearn_Option(N=128*128, M=30, K=2, Filter_Width=8, Lambda=2, Rho=1, coef_iteration=300, dict_iteration=2)
X = np.zeros((dictlearn_opt.K, dictlearn_opt.M, dictlearn_opt.N))

S_test = np.zeros((test_opt.K, test_opt.N), dtype=np.float)

S_test[0] = exim.image('sail.png', idxexp=np.s_[:, 210:722]).flatten()
S_test[1] = exim.image('tulips.png', idxexp=np.s_[:, 30:542]).flatten()
for i in range(test_opt.K):
    S_test[i] = util.normalize(S[i])

Coef_test = Coef_update_L1(test_opt, S_test)

# %%
X = Coef_test.coef_update_l1(D)

reconstruct = Coef_test.get_reconstruct()
for i in range(test_opt.K):
    plot.imview(reconstruct[i])

Coef_test.get_psnr_and_ssim()
Coef_test.get_L0()
# %%
