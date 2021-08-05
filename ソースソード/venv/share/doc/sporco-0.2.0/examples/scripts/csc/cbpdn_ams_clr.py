#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
CSC with a Spatial Mask
=======================

This example demonstrates the use of :class:`.cbpdn.AddMaskSim` for convolutional sparse coding with a spatial mask :cite:`wohlberg-2016-boundary`. The example problem is inpainting of randomly distributed corruption of a colour image :cite:`wohlberg-2016-convolutional`.
"""


from __future__ import print_function
from builtins import input

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

from sporco import util
from sporco import signal
from sporco import plot
from sporco.admm import tvl2
from sporco.admm import cbpdn
from sporco.metric import psnr


"""
Load a reference image.
"""

img = util.ExampleImages().image('monarch.png', zoom=0.5, scaled=True,
                                 idxexp=np.s_[:, 160:672])



"""
Create random mask and apply to reference image to obtain test image. (The call to ``numpy.random.seed`` ensures that the pseudo-random noise is reproducible.)
"""

np.random.seed(12345)
frc = 0.5
msk = signal.rndmask(img.shape, frc, dtype=np.float32)
imgw = msk * img


"""
Define pad and crop functions.
"""

pn = 8
spad = lambda x: np.pad(x, ((pn, pn), (pn, pn), (0, 0)), mode='symmetric')
zpad = lambda x: np.pad(x, ((pn, pn), (pn, pn), (0, 0)), mode='constant')
crop = lambda x: x[pn:-pn, pn:-pn]


"""
Construct padded mask and test image.
"""

mskp = zpad(msk)
imgwp = spad(imgw)


"""
$\ell_2$-TV denoising with a spatial mask as a non-linear lowpass filter. The highpass component is the difference between the test image and the lowpass component, multiplied by the mask for faster convergence of the convolutional sparse coding (see :cite:`wohlberg-2017-convolutional3`).
"""

lmbda = 0.05
opt = tvl2.TVL2Denoise.Options({'Verbose': False, 'MaxMainIter': 200,
                    'DFidWeight': mskp, 'gEvalY': False,
                    'AutoRho': {'Enabled': True}})
b = tvl2.TVL2Denoise(imgwp, lmbda, opt, caxis=2)
sl = b.solve()
sh = mskp * (imgwp - sl)


"""
Load dictionary.
"""

D = util.convdicts()['RGB:8x8x3x64']


"""
Set up :class:`.admm.cbpdn.ConvBPDN` options.
"""

lmbda = 2e-2
opt = cbpdn.ConvBPDN.Options({'Verbose': True, 'MaxMainIter': 200,
                    'HighMemSolve': True, 'RelStopTol': 5e-3,
                    'AuxVarObj': False, 'RelaxParam': 1.8,
                    'rho': 5e1*lmbda + 1e-1, 'AutoRho': {'Enabled': False,
                    'StdResiduals': False}})


"""
Construct :class:`.admm.cbpdn.AddMaskSim` wrapper for :class:`.admm.cbpdn.ConvBPDN` and solve via wrapper. This example could also have made use of :class:`.admm.cbpdn.ConvBPDNMaskDcpl`, which has similar performance in this application, but :class:`.admm.cbpdn.AddMaskSim` has the advantage of greater flexibility in that the wrapper can be applied to a variety of CSC solver objects.
"""

ams = cbpdn.AddMaskSim(cbpdn.ConvBPDN, D, sh, mskp, lmbda, opt=opt)
X = ams.solve()


"""
Reconstruct from representation.
"""

imgr = crop(sl + ams.reconstruct().squeeze())


"""
Display solve time and reconstruction performance.
"""

print("AddMaskSim wrapped ConvBPDN solve time: %.2fs" %
      ams.timer.elapsed('solve'))
print("Corrupted image PSNR: %5.2f dB" % psnr(img, imgw))
print("Recovered image PSNR: %5.2f dB" % psnr(img, imgr))


"""
Display reference, test, and reconstructed image
"""

fig = plot.figure(figsize=(21, 7))
plot.subplot(1, 3, 1)
plot.imview(img, title='Reference image', fig=fig)
plot.subplot(1, 3, 2)
plot.imview(imgw, title='Corrupted image', fig=fig)
plot.subplot(1, 3, 3)
plot.imview(imgr, title='Reconstructed image', fig=fig)
fig.show()


"""
Display lowpass component and sparse representation
"""

fig = plot.figure(figsize=(14, 7))
plot.subplot(1, 2, 1)
plot.imview(sl, cmap=plot.cm.Blues, title='Lowpass component', fig=fig)
plot.subplot(1, 2, 2)
plot.imview(np.squeeze(np.sum(abs(X), axis=ams.cri.axisM)),
            cmap=plot.cm.Blues, title='Sparse representation', fig=fig)
fig.show()


"""
Plot functional value, residuals, and rho
"""

its = ams.getitstat()
fig = plot.figure(figsize=(21, 7))
plot.subplot(1, 3, 1)
plot.plot(its.ObjFun, xlbl='Iterations', ylbl='Functional', fig=fig)
plot.subplot(1, 3, 2)
plot.plot(np.vstack((its.PrimalRsdl, its.DualRsdl)).T, ptyp='semilogy',
          xlbl='Iterations', ylbl='Residual', lgnd=['Primal', 'Dual'],
          fig=fig)
plot.subplot(1, 3, 3)
plot.plot(its.Rho, xlbl='Iterations', ylbl='Penalty Parameter', fig=fig)
fig.show()


# Wait for enter on keyboard
input()
