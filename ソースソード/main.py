# %%
import numpy as np
from tqdm import tqdm 
import Option
import util

import Convert
from Coef_update_L1 import Coef_update_L1
from Dict_update_L1 import Dict_update_L1
import Image_IO
import Encode 
from Decode_Fourier_L2 import Decode_Fourier_L2
from Decode_Random_L2 import Decode_Random_L2
from sporco import plot 
import matplotlib.pyplot as plt

#/////////////////////////////////////////////////////////
#辞書作成(L1-L1)
#/////////////////////////////////////////////////////////
# %%
'''
パラメータ設定
'''
np.random.seed(seed=123)
dictlearn_opt = Option.DictLearn_Option(N=128*128, M=10, Filter_Width=12, Lambda=0.01, Rho=1, coef_iteration=2, dict_iteration=2)
D = np.random.normal(0, 1, (dictlearn_opt.M, dictlearn_opt.Filter_Width, dictlearn_opt.Filter_Width))
D = util.padding(D, dictlearn_opt)
X = np.zeros((dictlearn_opt.M, dictlearn_opt.N))

KF = Image_IO.Image_In(1)

Coef = Coef_update_L1(dictlearn_opt, KF)
Dict = Dict_update_L1(dictlearn_opt, KF) 
Dict.D = D
Dict.get_tiled_D() 
#%%
'''
辞書学習
'''
for i in tqdm(range(100)):
    X = Coef.coef_update_l1(D)
    D = Dict.dict_update_l1(X)


Dict.get_tiled_D()

#%%
'''
作成した辞書の性能評価(リコンストラクトを作成してそれを評価)
'''
dictlearn_opt.coef_iteration = 500
dictlearn_opt.Lambda = 2
X = np.zeros((dictlearn_opt.M, dictlearn_opt.N))
X = Coef.coef_update_l1(D)
plot.imview(Coef.get_reconstruct())
Coef.get_L0()
Coef.get_psnr_and_ssim()


#/////////////////////////////////////////////////////////
#フーリエ行列を用いた観測行列での復号(L2-L1)
#/////////////////////////////////////////////////////////
# %%
'''
パラメータ設定
'''
decode_fourier_opt = Option.Decode_Fourier_Option(N=128*128, M=10, L=128*128 // 2, Myu=0.05, Rho=0.01, iteration=300, filter="low")
NKF = Image_IO.Image_In(2)

Decode_Fourier = Decode_Fourier_L2(decode_fourier_opt, NKF)
# %%
'''
ノンキーフレームを圧縮
'''
Y = Encode.encode_fourier(NKF, decode_fourier_opt)
# %%
'''
復号化
'''
decode_img = Decode_Fourier.decode(D, Y)
plot.imview(Decode_Fourier.get_decode_img(), title="Decode_Fourier")
Decode_Fourier.get_psnr_and_ssim()



#/////////////////////////////////////////////////////////
#ランダム行列を用いた観測行列での復号(L2-L1)
#/////////////////////////////////////////////////////////
# %%
'''
パラメータ設定
'''
decode_random_opt = Option.Decode_Random_Option(N=128*128, M=10, L=128*128 // 2, Myu=0.00001, Rho=50, iteration=100)
NKF = Image_IO.Image_In(2)

Decode_Random = Decode_Random_L2(decode_random_opt, NKF)
# %%
'''
ノンキーフレームを圧縮
'''
Y = Encode.encode_random(NKF, decode_random_opt)
# %%
'''
復号化
'''
decode_img = Decode_Random.decode(D, Y)
plot.imview(Decode_Random.get_decode_img(), title="Decode_Random")
Decode_Random.get_psnr_and_ssim()
# %%
