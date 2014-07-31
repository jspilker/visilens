import numpy as np
from Data_objs import Visdata

def box(u,v,du,dv):
      return 1.

def expsinc(u,v,du,dv):
      a1, a2 = 1.55, 2.52
      return np.sinc(u/(a1*du))*np.exp(-(u/(a2*du))**2.) * \
            np.sinc(v/(a1*du))*np.exp(-(v/(a2*dv))**2.)

def uvgrid(visdata,gridsize=256,binsize=2.0,convolution='box'):
      """
      Grid a set of given visibilities onto a grid of gridsize x gridsize.
      
      Inputs:
      visdata:
            Any visdata object.
      gridsize:
            Size of grid to be binned onto, NxN.
      binsize:
            Size of uv bin.
      convolution:
            Which convolution kernel to use. Can be either 'box' or 'expsinc'

      Returns:
      gridvis:
            The visibilities gridded as requested.
      """
      
      if not convolution.lower() in ['box','expsinc']:
            raise ValueError("Unrecognized convolution kernel; must be one of 'box','expsinc'")

      # Calculate the new u and v grids; create placeholders for the rest
      nu = (np.arange(gridsize) - gridsize/2.)*binsize
      newu,newv = np.meshgrid(nu,nu)
      newr = np.zeros((gridsize,gridsize))
      newi = np.zeros((gridsize,gridsize))
      news = np.zeros((gridsize,gridsize))

      i = np.round(visdata.u/binsize + gridsize/2. - 1)
      j = np.round(visdata.v/binsize + gridsize/2.)

      if convolution.lower() == 'box':
            conv_func, ninclude = box, 1
      elif convolution.lower() == 'expsinc':
            conv_func, ninclude = expsinc, 7

      incrange = np.arange(ninclude) - (ninclude-1)/2.
      for k in range(visdata.u.size):
            for l in incrange+j[k]:
                  for m in incrange+i[k]:
                        print k,l,m,j[k],i[k]
                        conv = conv_func(visdata.u[k]-newu[l,m],visdata.v[k]-newv[l,m],\
                                          binsize,binsize)
                        newr[l,m] += (visdata.real[k]*(visdata.sigma[k]**-2.)).sum()*conv
                        newi[l,m] += (visdata.imag[k]*(visdata.sigma[k]**-2.)).sum()*conv
                        news[l,m] += (visdata.sigma[k]**-2.).sum()**-0.5*conv

      newu = newu.reshape(gridsize**2)
      newv = newv.reshape(gridsize**2)
      newr = newr.reshape(gridsize**2)
      newi = newi.reshape(gridsize**2)
      news = news.reshape(gridsize**2)

      return visdata(newu,newv,newr,newi,news)
