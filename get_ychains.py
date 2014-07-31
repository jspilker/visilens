# Script on how to get info from yashar's chains
import sys
sys.path.append('/home/jspilker/Research')
import gravity as gl
from scipy.io import loadmat
import numpy as np

a = loadmat('0346_chain_matchnorescale.mat')

lnlike,lenspars,srcpars = [],[],[]
for arr in a['chain_1'][1:]:
      lnlike.append(arr[0])
      lenspars.append(arr[1])
      srcpars.append(arr[2])

lnlike = np.squeeze(np.asarray(lnlike))
lenspars = np.squeeze(np.asarray(lenspars))
srcpars = np.squeeze(np.asarray(srcpars))
allpars = np.c_[lenspars,srcpars]
allpars[:,4] *= -1
allpars[:,-1] *= -1
xL,yL,ML,eL,PAL = allpars[:,3],allpars[:,4],allpars[:,0],allpars[:,1],allpars[:,2]
xS,yS,FS,RS = allpars[:,7],allpars[:,8],allpars[:,5],allpars[:,6]
allpars = np.c_[xL,yL,ML,eL,PAL,xS,yS,FS,RS]
allpars[:,4] = 180.-allpars[:,4]

# a pseudo-burnin
lnlike = lnlike[5000:,:]
allpars = allpars[5000:,:]

chaindict = {}
cols = ['xL','yL','ML','eL','PAL','xoffS0','yoffS0','fluxS0','widthS0']
chaindict['chains'] = np.core.records.fromarrays(allpars.transpose(),names=cols)

f,axesarray = gl.TrianglePlot_MCMC(chaindict,plotmag=True,plotnuisance=True)
f.savefig('ycode_matchnorescale.png')

