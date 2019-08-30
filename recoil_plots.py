import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def recoil_plots(truth, pred, pfmet, puppi, path):
    px_truth, py_truth, pt_truth = truth
    px_pred, py_pred, pt_pred, par_pred = pred

    px_pred_pfmet, py_pred_pfmet, par_pred_pfmet = pfmet
    px_pred_puppi, py_pred_puppi, par_pred_puppi = puppi

    plt.style.use('default')
    plotrange = 100

    # PX
    plt.figure(figsize=(24, 6))
    plt.subplot(131)
    plt.xlabel('recoil px truth [GeV]')
    plt.ylabel('recoil px DeepRecoil [GeV]')
    #plt.scatter(px_truth, px_pred, s=0.25, c='k')
    plt.hist2d(px_truth, px_pred, norm=LogNorm(),
               bins=50, range=[[-plotrange, plotrange], [-plotrange, plotrange]])
    plt.colorbar()
    #pt_diff = (pt_pred - pt_truth)
    plt.subplot(132)
    plt.xlabel('recoil px [GeV]')
    plt.hist(px_truth, bins=50, range=(-plotrange, plotrange),
             histtype='step', label='truth')
    plt.hist(px_pred, bins=50, range=(-plotrange, plotrange),
             histtype='step', label='prediction DeepRecoil')
    plt.hist(px_pred_pfmet, bins=50, range=(-plotrange, plotrange),
             histtype='step', label='prediction pfmet')
    plt.hist(px_pred_puppi, bins=50, range=(-plotrange, plotrange),
             histtype='step', label='prediction hpuppimet')
    plt.legend(loc='upper right')
    plt.subplot(133)
    plt.xlabel('bias px [GeV]')
    plt.hist(px_truth, bins=50, range=(-plotrange, plotrange),
             histtype='step', label='bias 0')
    plt.hist(px_pred - px_truth, bins=50, range=(-plotrange, plotrange),
             histtype='step', label='bias DeepRecoil')
    plt.hist(px_pred_pfmet - px_truth, bins=50, range=(-plotrange,
                                                       plotrange), histtype='step', label='bias pfmet')
    plt.hist(px_pred_puppi - px_truth, bins=50, range=(-plotrange,
                                                       plotrange), histtype='step', label='bias hpuppimet')
    plt.legend(loc='upper right')

    plt.savefig('%s/px.pdf' % path, bbox_inches='tight')


    # In[155]:


    # PY
    plt.figure(figsize=(24, 6))
    plt.subplot(131)
    plt.xlabel('recoil py truth [GeV]')
    plt.ylabel('recoil py DeepRecoil [GeV]')
    #plt.scatter(py_truth, py_pred, s=0.25, c='w')
    plt.hist2d(py_truth, py_pred, norm=LogNorm(),
               bins=50, range=[[-plotrange, plotrange], [-plotrange, plotrange]])
    plt.colorbar()
    #pt_diff = (pt_pred - pt_truth)
    plt.subplot(132)
    plt.xlabel('recoil py [GeV]')
    plt.hist(py_truth, bins=50, range=(-plotrange, plotrange),
             histtype='step', label='truth')
    plt.hist(py_pred, bins=50, range=(-plotrange, plotrange),
             histtype='step', label='prediction DeepRecoil')
    plt.hist(py_pred_pfmet, bins=50, range=(-plotrange, plotrange),
             histtype='step', label='prediction pfmet')
    plt.hist(py_pred_puppi, bins=50, range=(-plotrange, plotrange),
             histtype='step', label='prediction hpuppimet')
    plt.legend(loc='upper right')
    plt.subplot(133)
    plt.xlabel('bias py [GeV]')
    plt.hist(py_truth, bins=50, range=(-plotrange, plotrange),
             histtype='step', label='bias 0')
    plt.hist(py_pred - py_truth, bins=50, range=(-plotrange, plotrange),
             histtype='step', label='bias DeepRecoil')
    plt.hist(py_pred_pfmet - py_truth, bins=50, range=(-plotrange,
                                                       plotrange), histtype='step', label='bias pfmet')
    plt.hist(py_pred_puppi - py_truth, bins=50, range=(-plotrange,
                                                       plotrange), histtype='step', label='bias hpuppimet')
    plt.legend(loc='upper right')

    plt.savefig('%s/py.pdf' % path, bbox_inches='tight')


    # PT
    plt.figure(figsize=(14, 6))
    plt.subplot(121)
    plt.xlabel('Z pT truth [GeV]')
    plt.ylabel('Z pT DeepRecoil [GeV]')
    plt.hist2d(pt_truth, pt_pred, norm=LogNorm(),
               bins=50, range=[[0, 2*plotrange], [0, 2*plotrange]])
    plt.colorbar()
    #pt_diff = (pt_pred - pt_truth)
    plt.subplot(122)
    plt.xlabel('Z pT bias [GeV]')
    plt.hist(pt_truth, bins=50, range=(-plotrange, plotrange),
             histtype='step', label='truth')
    plt.hist(pt_pred, bins=50, range=(-plotrange, plotrange),
             histtype='step', label='prediction DeepRecoil')
    plt.hist(pt_pred - pt_truth, bins=50, range=(-plotrange, plotrange),
             histtype='step', label='bias DeepRecoil')
    plt.legend(loc='upper right')

    plt.savefig('%s/pt.pdf' % path, bbox_inches='tight')


    # UPAR
    plt.figure(figsize=(30, 6))
    plt.subplot(141)
    plt.xlabel('Z pT truth [GeV]')
    plt.ylabel('$u_{||}^W$ CNN [GeV]')
    x = np.arange(0, 1*plotrange, 0.1)
    plt.plot(x, x, color='red')
    plt.hist2d(pt_truth, par_pred, norm=LogNorm(),
               bins=50, range=[[0., 1*plotrange], [-0.5*plotrange, 1.5*plotrange]])
    plt.colorbar()

    plt.subplot(142)
    plt.xlabel('Z pT truth [GeV]')
    plt.ylabel('$u_{||}^W$ CNN - Z pT truth [GeV]')
    x = np.arange(0, 3*plotrange, 0.1)
    y = x*0
    plt.plot(x, y, color='red')
    plt.hist2d(pt_truth, par_pred-pt_truth, norm=LogNorm(),
               bins=50, range=[[0., 1*plotrange], [-plotrange, plotrange]])
    plt.colorbar()

    plt.subplot(143)
    plt.xlabel('Z pT CNN [GeV]')
    plt.ylabel('$u_{||}^W$ CNN - Z pT truth [GeV]')
    x = np.arange(0, 3*plotrange, 0.1)
    y = x*0
    plt.plot(x, y, color='red')
    plt.hist2d(pt_pred, par_pred-pt_truth, norm=LogNorm(),
               bins=50, range=[[0., 1*plotrange], [-plotrange, plotrange]])
    plt.colorbar()

    #pt_diff = (pt_pred - pt_truth)
    plt.subplot(144)
    plt.xlabel('$u_{||}^W$ bias [GeV]')
    plt.hist(-pt_truth, bins=50, range=(-plotrange, plotrange),
             histtype='step', label='bias 0')
    plt.hist(par_pred - pt_truth, bins=50, range=(-plotrange,
                                                  plotrange), histtype='step', label='bias CNN')
    plt.hist(par_pred_pfmet - pt_truth, bins=50, range=(-plotrange,
                                                        plotrange), histtype='step', label='bias pfmet')
    plt.hist(par_pred_puppi - pt_truth, bins=50, range=(-plotrange,
                                                        plotrange), histtype='step', label='bias hpuppimet')
    plt.hist(pt_truth, bins=50, range=(-plotrange, plotrange),
             histtype='step', label='truth', linestyle='--', color='blue')
    plt.hist(par_pred, bins=50, range=(-plotrange, plotrange),
             histtype='step', label='prediction CNN', linestyle='--', color='orange')
    plt.legend(loc='upper right')

    plt.savefig('%s/upar.pdf' % path, bbox_inches='tight')
