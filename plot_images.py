import numpy as np
from scipy.ndimage.measurements import center_of_mass
from astropy.stats import sigma_clip
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from Model_objs import *
from Data_objs import *
from uvimage import uvimageslow
from calc_likelihood import create_modelimage,fft_interpolate
from modelcal import model_cal
from GenerateLensingGrid import GenerateLensingGrid

def plot_images(data,mcmcresult,returnimages=False,
                  imsize=512,pixsize=0.2,taper=0.,**kwargs):
      """
      Create a four-panel figure from data and chains,
      showing data, best-fit model, residuals, high-res image.

      Inputs:
      data:
            The visdata object(s) to be imaged. If more than
            one is passed, they'll be imaged/differenced separately.
      chains:
            A result from running LensModelMCMC.

      returnimages: bool, optional
            If True, will also return a list of numpy arrays
            containing the imaged data, interpolated model, and full-res model.
            Default is False.

      Returns:
      If returnimages is False:
      f, axarr:
            A matplotlib figure and array of Axes objects.

      If returnimages is True:
      f,axarr,imagelist:
            Same as above; imagelist is a list of length (# of datasets),
            containing three arrays each, representing the data,
            interpolated model, and full-resolution model.
      """

      limits = kwargs.pop('limits',
            [-imsize*pixsize/2.,+imsize*pixsize/2.,imsize*pixsize/2.,-imsize*pixsize/2.])
      cmap = kwargs.pop('cmap',cm.Greys)
      mapcontours = kwargs.pop('mapcontours',np.delete(np.arange(-21,22,3),7))
      rescontours = kwargs.pop('rescontours',np.array([-6,-5,-4,-3,-2,2,3,4,5,6]))

      datasets = list(np.array([data]).flatten())

      # shorthand for later
      c = mcmcresult['chains']

      # Set up to create the model image. We'll assume the best-fit values are all the medians.
      lens,source = mcmcresult['lens_p0'], mcmcresult['source_p0']
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
                  elif t: scaleamp[i] = np.median(c['ampscale_dset'+str(i)])
                  else: pass
      shiftphase = np.zeros((len(datasets),2))
      if 'shiftphase' in mcmcresult.keys():
            for i,t in enumerate(mcmcresult['shiftphase']):
                  if i==0: pass # only matters if >1 datasets
                  elif t:
                        shiftphase[i][0] = np.median(c['astromshift_x_dset'+str(i)])
                        shiftphase[i][1] = np.median(c['astromshift_y_dset'+str(i)])
                  else: pass # no shifting

      sourcedatamap = mcmcresult['sourcedatamap'] if 'sourcedatamap' in mcmcresult.keys() else None
      modelcal = mcmcresult['modelcal'] if 'modelcal' in mcmcresult.keys() else [False]*len(datasets)

      f,axarr = pl.subplots(len(datasets),4,figsize=(12,3*len(datasets)))
      axarr = np.atleast_2d(axarr)
      images = [[] for _ in range(len(datasets))] # effing mutable lists.
      
      for i,dset in enumerate(datasets):
            # Get us some coordinates.
            xmap,ymap,xemit,yemit,ix = GenerateLensingGrid(datasets,mcmcresult['xmax'],\
                  mcmcresult['highresbox'],mcmcresult['fieldres'],mcmcresult['emitres'])
            
            # Create model image
            immap,_ = create_modelimage(lens,source,shear,xmap,ymap,xemit,yemit,\
                  ix,sourcedatamap)

            # And interpolate onto uv-coords of dataset
            interpdata = fft_interpolate(dset,immap,xmap,ymap,ug=None,\
                  scaleamp=scaleamp[i],shiftphase=shiftphase[i])

            if modelcal[i]: interpdata,_ = model_cal(dset,interpdata)
            
            # Image the data
            imdata = uvimageslow(dset,imsize,pixsize,taper)
            # Image the model
            immodel = uvimageslow(interpdata,imsize,pixsize,taper)
            # And the residuals
            imdiff = imdata - immodel

            if returnimages: 
                  images[i].append(imdata); images[i].append(immodel)#; images[i].append(immap)
            
            # Plot everything up
            ext = [-imsize*pixsize/2.,imsize*pixsize/2.,imsize*pixsize/2.,-imsize*pixsize/2.]
            #s = (sigma_clip(imdata.flatten(),sig=2.5,iters=None)).std() # Map noise, roughly.
            s = ((dset.sigma**-2.).sum())**-0.5 # Map noise, roughly.
            #s = imdiff.std() # Map noise, roughly.
            print s
            axarr[i,0].imshow(imdata,interpolation='nearest',extent=ext,cmap=cmap)
            axarr[i,0].contour(imdata,extent=ext,colors='k',origin='image',levels=s*mapcontours)
            axarr[i,0].set_xlim(limits[0],limits[1]); axarr[i,0].set_ylim(limits[2],limits[3])
            axarr[i,1].imshow(immodel,interpolation='nearest',extent=ext,cmap=cmap,\
                  vmin=imdata.min(),vmax=imdata.max())
            axarr[i,1].contour(immodel,extent=ext,colors='k',origin='image',levels=s*mapcontours)
            axarr[i,1].set_xlim(limits[0],limits[1]); axarr[i,1].set_ylim(limits[2],limits[3])
            axarr[i,2].imshow(imdiff,interpolation='nearest',extent=ext,cmap=cmap,\
                  vmin=imdata.min(),vmax=imdata.max())
            axarr[i,2].contour(imdiff,extent=ext,colors='k',origin='image',levels=s*rescontours)
            axarr[i,2].set_xlim(limits[0],limits[1]); axarr[i,2].set_ylim(limits[2],limits[3])
            if np.log10(s) < -6.: sig,unit = 1e9*s,'nJy'
            elif np.log10(s) < -3.: sig,unit = 1e6*s,'$\mu$Jy'
            elif np.log10(s) < 0.: sig,unit = 1e3*s,'mJy'
            else: sig,unit = s,'Jy'
            axarr[i,2].text(0.1,0.1,"1$\sigma$ = {0:.1f}{1:s}".format(sig,unit),
                  transform=axarr[i,2].transAxes,bbox=dict(fc='w'))
            #axarr[i,3].imshow(immap,interpolation='nearest',\
            #      extent=[xmap.min(),xmap.max(),xmap.max(),xmap.min()],cmap=cmap)

            # Give a zoomed-in view in the last panel
            # Create model image at higher res
            imemit,_ = create_modelimage(lens,source,shear,xemit,yemit,xemit,yemit,\
                  [0,xemit.shape[1],0,xemit.shape[0]],sourcedatamap)

            images[i].append(imemit)

            axarr[i,3].imshow(imemit,interpolation='nearest',\
                  extent=[xemit.min(),xemit.max(),yemit.max(),yemit.min()],cmap=cmap)
            
            xcen = center_of_mass(imemit)[1]*(xemit[0,1]-xemit[0,0]) + xemit.min()
            ycen = center_of_mass(imemit)[0]*(xemit[0,1]-xemit[0,0]) + yemit.min()
            dx = 0.6*(xemit.max()-xemit.min())
            dy = 0.6*(yemit.max()-yemit.min())
            axarr[i,3].set_xlim(xcen-dx,xcen+dx); axarr[i,3].set_ylim(ycen+dy,ycen-dy)
            
            #axarr[i,3].set_xlim(xemit.min(),xemit.max())
            #axarr[i,3].set_ylim(yemit.max(),yemit.min()+1.)
            # Label some axes and such
            axarr[i,0].set_title(dset.filename+'\nDirty Image')
            axarr[i,1].set_title('Model Dirty Image')
            axarr[i,2].set_title('Residuals')
            axarr[i,3].set_title('High-res Model')
            
      

      if returnimages: return f,axarr,images
      else: return f,axarr
      
