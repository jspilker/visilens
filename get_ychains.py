# Script on how to get info from yashar's chains
import sys
sys.path.append('/home/jspilker/Research')
import visilens as vl
from scipy.io import loadmat
import numpy as np
arcsec2rad = np.pi/180./3600.

a = loadmat('mcmc_chain.mat')

lnlike,lenspars,srcpars = [],[],[]
for arr in a['chain_1'][1:]:
      lnlike.append(arr[0])
      lenspars.append(arr[1])
      srcpars.append(arr[2])

lnlike = np.squeeze(np.asarray(lnlike))
lenspars = np.squeeze(np.asarray(lenspars))
srcpars = np.squeeze(np.asarray(srcpars))
allpars = np.c_[lenspars,srcpars]
#allpars[:,4] *= -1
#allpars[:,-1] *= -1
xL,yL,ML,eL,PAL = allpars[:,3],allpars[:,4],allpars[:,0],allpars[:,1],allpars[:,2]
xS,yS,FS,RS = allpars[:,7],allpars[:,8],allpars[:,5],allpars[:,6]
allpars = np.c_[xL,yL,ML,eL,PAL,xS,yS,FS,RS]
#allpars[:,4] = 180.-allpars[:,4]

# a pseudo-burnin
lnlike = lnlike[10000:]
allpars = allpars[10000:,:]

chaindict = {}
cols = ['xL','yL','ML','eL','PAL','xoffS0','yoffS0','fluxS0','widthS0']
chaindict['chains'] = np.core.records.fromarrays(allpars.transpose(),names=cols)
chaindict['lnlike'] = lnlike

f,axesarray = vl.TrianglePlot_MCMC(chaindict,plotmag=True,plotnuisance=True)
f.savefig('yasharchains_triangle.png')

fname = 'ALMA_2_unselcald.mat'
a = loadmat(fname)
comdata = vl.Visdata(a['u'].flatten(),a['v'].flatten(),a['vis_data'].flatten().real,a['vis_data'].flatten().imag,\
            a['sig'].flatten(),a['ant1'].flatten(),a['ant2'].flatten())
#comdata = gl.visdata(comdata.u,comdata.v,a['vis_data'].flatten().real,a['vis_data'].flatten().imag,\
#            a['sig'].flatten(),a['ant1'].flatten(),a['ant2'].flatten())

comdata.sigma *= 0.0203
comdata.filename = fname
comdata.PBfwhm = 21.0 # arcsec

# All the baselines to ant12 are flagged (0 amp), remove them
good = np.where((comdata.ant1!=12) & (comdata.ant2!=12))
comdata = vl.Visdata(comdata.u[good],comdata.v[good],comdata.real[good],comdata.imag[good],comdata.sigma[good],\
            comdata.ant1[good],comdata.ant2[good],filename=comdata.filename,PBfwhm=comdata.PBfwhm)


chaindict['lens_p0'] = vl.SIELens(0.263,xL.mean(),yL.mean(),ML.mean(),eL.mean(),PAL.mean())
chaindict['source_p0'] = [vl.GaussSource(4.224,True,xS.mean(),yS.mean(),FS.mean(),RS.mean())]
chaindict['xmax'] = 40.
chaindict['highresbox'] = [-7, -2.5, 1.,5.5]
chaindict['emitres'] = 0.015
chaindict['fieldres'] = (2*4*comdata.uvdist.max())**-1 / arcsec2rad
chaindict['modelcal'] = [True]
chaindict['data'] = [comdata.filename]

f,axarr,ims = vl.plot_images(comdata,chaindict,imsize=250,pixsize=0.2,returnimages=True,\
      mapcontours=np.array([-28,-23,-18,-13,-8,-3,3,8,13,18,23,28,33]),\
      rescontours=np.array([-6,-5,-4,-3,-2,-1,1,2,3,4,5,6]),level=0.0017,limits=[-15,5,13,-7])
f.savefig('yasharchains_panels.png')

