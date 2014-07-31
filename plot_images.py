import numpy as np
import matplotlib.pyplot as pl
from uvimage import uvimageslow

def plot_images(data,chains,**kwargs):
      """
      Create a four-panel figure from data and chains,
      showing data, best-fit model, residuals, high-res image.

      Inputs:
      data:
            The visdata object(s) to be imaged. If more than
            one is passed, they'll be concatenated and imaged
            jointly.
      chains:
            A result from running LensModelMCMC.
      **kwargs:
            Various keywork arguments are possible, most likely
            are imsize and pixsize to be passed to imaging. Also
            can pass extent=[xmin, xmax, ymin, ymax] to set axis
            limits on the four plots.

      Returns:
      f, axarr:
            A matplotlib figure and array of Axes objects.
      """

      imsize = kwargs.pop('imsize',512)
      pixsize = kwargs.pop('pixsize',0.2)
      extent = kwargs.pop('extent',
            [-imsize*pixsize/2.,+imsize*pixsize/2.,-imsize*pixsize/2.,+imsize*pixsize/2.])

      f,axarr = pl.subplots(1,4,figsize=(12,3))

      datasets = list(np.array([data]).flatten())
      imdata = np.array(np.zeros(imsize,imsize),dtype=complex)      
      for dset in datasets:
            # Check to see if conjugate points already taken care of
            if np.all(dset.u[:dset.u.size/2] == -dset.u[dset.u.size/2:]):
                  imdata += uvimageslow(dset,imsize,pixsize)
            else:
                  imdata += 2.*(uvimageslow(dset,imsize,pixsize)).real

      # Set up to create the model image... probably should have just made this
      # its own routine instead of wrapping it up in calc_likelihood... ah well.
      zL,zS = chains['lens_p0'].z, chains['source_p0'][0].z
      xL,yL,ML = chains['chains']['xL'].mean(),chains['chains']['yL'].mean(),chains['chains']['ML'].mean()
      eL,PAL = chains['chains']['eL'].mean(),chains['chains']['PAL'].mean()
      lens = SIELens(zL,xL,yL,ML,eL,PAL)
      source = []
      for i,src in enumerate(chains['source_p0']):
            if isinstance(src,GaussSource):
                  source.append(GaussSource(zS,chains['chains']['xoffS'+str(i)].mean(),
                        chains['chains']['yoffS'+str(i)].mean(),chains['chains']['fluxS'+str(i)].mean(),
                        chains['chains']['widthS'+str(i)].mean()))

      if 'shear' in chains['chains'].dtype.names:
            shear = ExternalShear(chains['chains']['shear'].mean(),chains['chains']['shearangle'].mean())
      else: shear = None

      

def model_lens(data,lens,source,shear=None,imsize=256,pixsize=0.5,uvsample=False):
      """
      
      
