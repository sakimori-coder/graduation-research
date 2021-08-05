import numpy as np
import Option

# def FFT_NMKvector(f, opt):
#     rootN = int(np.sqrt(opt.N))
#     F = np.fft.fftn(f.reshape(opt.K, opt.M, rootN, -1), axes=(2,3))
#     return F.reshape(opt.K, opt.M, -1) / rootN

# def IFFT_NMKvector(F, opt):
#     rootN = int(np.sqrt(opt.N))
#     f = np.fft.ifftn(F.reshape(opt.K, opt.M, rootN, -1), axes=(2,3)) 
#     return f.reshape(opt.K, opt.M, -1) * rootN
    

# def FFT_NMvector(f, opt):
#     rootN = int(np.sqrt(opt.N))
#     F = np.fft.fftn(f.reshape(opt.M, rootN, -1), axes=(1,2))
#     return F.reshape(opt.M, -1) / rootN

# def IFFT_NMvector(F, opt):
#     rootN = int(np.sqrt(opt.N))
#     f = np.fft.ifftn(F.reshape(opt.M, rootN, -1), axes=(1,2)) 
#     return f.reshape(opt.M, -1) * rootN
    

# def FFT_D(D, opt):
#     rootN = int(np.sqrt(opt.N))
#     Df = np.fft.fftn(D.reshape(opt.M, rootN, -1), axes=(1,2))
#     return Df.reshape(opt.M, -1)
     
# def FFT_Nvector(f, opt):
#     rootN = int(np.sqrt(opt.N))
#     return np.fft.fftn(f.reshape(rootN, -1)).flatten() / rootN

# def IFFT_Nvector(F, opt):
#     rootN = int(np.sqrt(opt.N))
#     return np.fft.ifft(F.reshape(rootN, -1)).flatten() * rootN
     
# def FFT_NKvector(f, opt):
#     rootN = int(np.sqrt(opt.N))
#     F = np.fft.fftn(f.reshape(opt.K, rootN, -1), axes=(1,2))
#     return F.reshape(opt.K, -1) / rootN

# def IFFT_NKvector(F, opt):
#     rootN = int(np.sqrt(opt.N))
#     f = np.fft.ifftn(F.reshape(opt.K, rootN, -1), axes=(1,2)) 
#     return f.reshape(opt.K, -1) * rootN

# def FFT_X(X, opt):
#     rootN = int(np.sqrt(opt.N))
#     Xf = np.fft.fftn(X.reshape(opt.K, opt.M, rootN, -1), axes=(2,3))
#     return Xf.reshape(opt.K, opt.M, -1)

def FFT_NMKvector(f, opt):
    rootN = int(np.sqrt(opt.N))
    F = np.fft.fft(f, axis=2)
    return F / rootN

def IFFT_NMKvector(F, opt):
    rootN = int(np.sqrt(opt.N))
    f = np.fft.ifft(F, axis=2) 
    return f * rootN
    

def FFT_NMvector(f, opt):
    rootN = int(np.sqrt(opt.N))
    F = np.fft.fft(f, axis=1)
    return F / rootN

def IFFT_NMvector(F, opt):
    rootN = int(np.sqrt(opt.N))
    f = np.fft.ifft(F, axis=1) 
    return f * rootN
    

def FFT_D(D, opt):
    rootN = int(np.sqrt(opt.N))
    Df = np.fft.fft(D, axis=1)
    return Df
     
def FFT_Nvector(f, opt):
    rootN = int(np.sqrt(opt.N))
    return np.fft.fft(f) / rootN

def IFFT_Nvector(F, opt):
    rootN = int(np.sqrt(opt.N))
    return np.fft.ifft(F) * rootN
     
def FFT_NKvector(f, opt):
    rootN = int(np.sqrt(opt.N))
    F = np.fft.fft(f, axis=1)
    return F / rootN

def IFFT_NKvector(F, opt):
    rootN = int(np.sqrt(opt.N))
    f = np.fft.ifft(F, axis=1) 
    return f * rootN

def FFT_X(X, opt):
    rootN = int(np.sqrt(opt.N))
    Xf = np.fft.fft(X, axis=2)
    return Xf