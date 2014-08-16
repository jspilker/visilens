import numpy as np
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from uvimage import uvimageslow
from calc_likelihood import create_modelimage,fft_interpolate
from modelcal import model_cal
from GenerateLensingGrid import GenerateLensingGrid

def plot_images(data,chains,xmax=30.,highresbox=[-3.,3.,-3.,3.],emitres=None,fieldres=None,
                  imsize=512,pixsize=0.2,taper=0.):
      """
      Create a four-panel figure from data and chains,
      showing data, best-fit model, residuals, high-res image.

      Inputs:
      data:
            The visdata object(s) to be imaged. If more than
            one is passed, they'll be imaged/differenced separately.
      chains:
            A result from running LensModelMCMC.

      Returns:
      f, axarr:
            A matplotlib figure and array of Axes objects.
      """

      limits = kwargs.pop('limits',
            [-imsize*pixsize/2.,+imsize*pixsize/2.,imsize*pixsize/2.,-imsize*pixsize/2.])
      cmap = kwargs.pop('cmap',cm.Greys)

      datasets = list(np.array([data]).flatten())

      # shorthand for later
      c = chains['chains']

      # Set up to create the model image. We'll assume the best-fit values are all the medians.
      lens,source = chains['lens_p0'], chains['source_p0']
      if isinstance(lens,SIELens):
            for key in ['x','y','M','e','PA']:
                  if not vars(lens)[key]['fixed']:
                        lens.__dict__[key]['value'] = np.median(c[key+'L'])
      # now do the source(s)
      for i,src in enumerate(source): # Source is a list of source objects
            if isinstance(src,GaussSource):
                  for key in ['xoff','yoff','flux','width']:
                        if not vars(src)[key]['fixed']:
                              src.__dict__[key]['value'] = np.median(c[key+'S'+str(i)])
            elif isinstance(src,SersicSource):
                  for key in ['xoff','yoff','flux','alpha','index','axisratio','PA']:
                        if not vars(src)[key]['fixed']:
                              src.__dict__[key]['value'] = np.median(c[key+'S'+str(i)])
      
      # now do shear, if any
      if 'shear_p0' in mcmcresult.keys():
            shear = mcmcresult['shear_p0']
            for key in ['shear','shearangle']:
                  if not vars(shear)[key]['fixed']:
                        shear.__dict__[key]['value'] = np.median(c[key])
      else: shear = None 

      # Any amplitude scaling or astrometric shifts
      scaleamp = np.ones(len(datasets))
      if 'scaleamp' in mcmcresult.keys():
            for i,t in enumerate(mcmcresult['scaleamp']): # only matters if >1 datasets
                  if i==0: pass
                  elif t: scaleamp[i] = np.median(c['ampscale_dset'+str(i+1)])
                  else: pass
      shiftphase = np.zeros((len(datasets),2))
      if 'shiftphase' in mcmcresult.keys():
            for i,t in enumerate(mcmcresult['shiftphase']):
                  if i==0: pass # only matters if >1 datasets
                  elif t:
                        shiftphase[i][0] = np.median(c['astromshift_x_dset'+str(i+1)])
                        shiftphase[i][1] = np.median(c['astromshift_y_dset'+str(i+1)])
                  else: pass # no shifting

      sourcedatamap = mcmcresult['sourcedatamap'] if 'sourcedatamap' in mcmcresult.keys() else None
      modelcal = mcmcresult['modelcal'] if 'modelcal' in mcmcresult.keys() else [False]*len(datasets)

      f,axarr = pl.subplots(len(datasets),4,figsize=(12,3*len(data)))
      
      for i,dset in enumerate(datasets):
            # Get us some coordinates.
            if isinstance(dset,Visdata):
                  xmap,ymap,xemit,yemit,ix = GenerateLensingGrid(datasets,mcmcresult['xmax'],\
                        mcmcresult['highresbox'],mcmcresult['fieldres'],mcmcresult['emitres'])
            
            # Create model image
            immap,_ = create_modelimage(lens,source,shear,xmap,ymap,xemit,yemit,\
                  ix,sourcedatamap)

            # And interpolate onto uv-coords of dataset
            interpdata = fft_interpolate(dset,immap,xmap,ymap,ug=None,\
                  scaleamp=scaleamp[i],shiftphase=shiftphase[i])

            if modelcal[i]: interpdata,_ = model_cal(dset,interpdata)
      

      
      
