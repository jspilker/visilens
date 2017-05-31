import numpy as np
import copy
import warnings
from scipy.fftpack import fftshift,fft2
from scipy.ndimage.measurements import center_of_mass
import matplotlib.pyplot as pl; pl.ioff()
import matplotlib.cm as cm
from matplotlib.colors import SymLogNorm
from class_utils import *
from calc_likelihood import *
from utils import *
from lensing import *

arcsec2rad = np.pi/180./3600.
c = 2.99792458e8 # in m/s

__all__ = ['uvimageslow','plot_images']

def uvimageslow(visdata,imsize=256,pixsize=0.5,taper=0.):
      """
      Invert a set of visibilities to the image plane in the slowest way possible
      (ie, no gridding of visibilities, just straight-up summation). This gets
      pretty darn close to imaging with CASA with natural weighting, no cleaning.
      
      Note: this is very slow, because it's very stupid (no gridding of visibilities
      or anything). If you need to make large images of a lot of data, I strongly
      recommend outputting your data to uvfits and imaging elsewhere, because I don't
      particularly want to re-create all of CASA's :clean: functionality on my own.
      
      Inputs:
      visdata:
            Any Visdata object
      imsize:
            Size of the output image in pixels
      pixsize:
            Pixel size, in arcseconds
      taper:
            Apply an additional gaussian taper to the
            visibilities with sigma of this, in arcsec.

      Returns:
      image:
            A 2D array containing the image of the inverted visibilities
      """

      thisvis = copy.deepcopy(visdata)

      if taper > 0.:
            uvsig = (taper * arcsec2rad)**-1.
            scalefac = np.exp(-thisvis.uvdist**2/(2*uvsig**2.))
            thisvis.sigma /= scalefac

      x = np.linspace(-imsize*pixsize/2. * arcsec2rad,+imsize*pixsize/2. * arcsec2rad,imsize)
      x,y = np.meshgrid(x,x)
      # These offsets line us up with CASA's coord system
      x -= pixsize/2. * arcsec2rad
      y += pixsize/2. * arcsec2rad

      im = np.array(np.zeros(x.shape),dtype=complex)

      # Check to see if we have the conjugate visibilities, and image.
      if np.all(thisvis.u[:thisvis.u.size/2] == -thisvis.u[thisvis.u.size/2:]):
            for i in range(thisvis.u.size/2):
                  im += (thisvis.sigma[i]**-2)*(thisvis.real[i]+1j*thisvis.imag[i]) *\
                        np.exp(2*np.pi*1j*((thisvis.u[i]*x)+(thisvis.v[i]*y)))
      else:
            for i in range(thisvis.u.size):
                  im += (thisvis.sigma[i]**-2)*(thisvis.real[i]+1j*thisvis.imag[i]) *\
                        np.exp(2*np.pi*1j*((thisvis.u[i]*x)+(thisvis.v[i]*y)))
            
      # We only imaged the non-conjugate visibilities; fix and renormalize to Jy/beam units
      im = 2*im.real
      im /= (2*thisvis.sigma**-2).sum()

      return im

def plot_images(data,mcmcresult,returnimages=False,plotcombined=False,plotall=False,
                  imsize=256,pixsize=0.2,taper=0.,**kwargs):
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
      
      plotcombined: bool, optional, default False
            If True, plots only a single row of images, after combining all datasets
            in `data' and applying any necessary rescaling/shifting contained in
            `mcmcresult'.
      
      plotall: bool, optional, default False
            If True, plots each dataset in its own row *and* the combined dataset in
            an extra row.

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
            [-imsize*pixsize/2.,+imsize*pixsize/2.,-imsize*pixsize/2.,+imsize*pixsize/2.])
      cmap = kwargs.pop('cmap',cm.Greys)
      mapcontours = kwargs.pop('mapcontours',np.delete(np.arange(-21,22,3),7))
      rescontours = kwargs.pop('rescontours',np.array([-6,-5,-4,-3,-2,2,3,4,5,6]))
      level = kwargs.pop('level',None)
      logmodel = kwargs.pop('logmodel',False)

      datasets = list(np.array([data]).flatten())

      # shorthand for later
      c = copy.deepcopy(mcmcresult['chains'])

      # Now all these things are saved separately, don't have to do the BS above
      lens = mcmcresult['best-fit']['lens']
      source=mcmcresult['best-fit']['source']
      scaleamp = mcmcresult['best-fit']['scaleamp'] if 'scaleamp' in mcmcresult['best-fit'].keys() else np.ones(len(datasets))
      shiftphase=mcmcresult['best-fit']['shiftphase'] if 'shiftphase' in mcmcresult['best-fit'].keys() else np.zeros((len(datasets),2))
      sourcedatamap = mcmcresult['sourcedatamap'] if 'sourcedatamap' in mcmcresult.keys() else [None]*len(datasets)
      modelcal = mcmcresult['modelcal'] if 'modelcal' in mcmcresult.keys() else [False]*len(datasets)
      
      
      if plotall:
            f,axarr = pl.subplots(len(datasets)+1,4,figsize=(14,4*(len(datasets)+1)))
            axarr = np.atleast_2d(axarr)
            images = [[] for _ in range(len(datasets)+1)]
            if sourcedatamap[0] is not None: warnings.warn("sourcedatamap[0] is not None. Are you sure you want plotall=True?")
            sourcedatamap.append(None)
      elif plotcombined:
            f,axarr = pl.subplots(1,4,figsize=(12,3))
            axarr = np.atleast_2d(axarr)
            images = [[]]
            sourcedatamap = [None]
      else:
            f,axarr = pl.subplots(len(datasets),4,figsize=(14,4*len(datasets)))
            axarr = np.atleast_2d(axarr)
            images = [[] for _ in range(len(datasets))] # effing mutable lists.
            
      plotdata,plotinterp = [],[]
      
      for i,dset in enumerate(datasets):
            # Get us some coordinates.
            xmap,ymap,xemit,yemit,ix = GenerateLensingGrid(datasets,mcmcresult['xmax'],\
                  mcmcresult['highresbox'],mcmcresult['fieldres'],mcmcresult['emitres'])
            
            # Create model image
            immap,_ = create_modelimage(lens,source,xmap,ymap,xemit,yemit,\
                  ix,sourcedatamap=sourcedatamap[i])

            # And interpolate onto uv-coords of dataset
            interpdata = fft_interpolate(dset,immap,xmap,ymap,ug=None,\
                  scaleamp=scaleamp[i],shiftphase=shiftphase[i])

            if modelcal[i]: 
                  selfcal,_ = model_cal(dset,interpdata)
            
            else: 
                  selfcal = copy.deepcopy(dset)
            
            plotdata.append(selfcal); plotinterp.append(interpdata)
            
      
      if plotall:
            plotdata.append(concatvis(plotdata))
            plotinterp.append(concatvis(plotinterp))      
      elif plotcombined: 
            plotdata = [concatvis(plotdata)]
            plotinterp = [concatvis(plotinterp)]

                  
      for row in range(axarr.shape[0]):
            
            # Image the data
            imdata = uvimageslow(plotdata[row],imsize,pixsize,taper)
            # Image the model
            immodel = uvimageslow(plotinterp[row],imsize,pixsize,taper)
            # And the residuals
            imdiff = imdata - immodel

            if returnimages: 
                  images[row].append(imdata); images[row].append(immodel)
            
            # Plot everything up
            ext = [-imsize*pixsize/2.,imsize*pixsize/2.,-imsize*pixsize/2.,imsize*pixsize/2.]
            # Figure out what to use as the noise level; sum of weights if no user-supplied value
            if level is None: s = ((plotdata[row].sigma**-2.).sum())**-0.5
            else:
                  try:
                        s = [e for e in level][i]
                  except TypeError:
                        s = float(level)
            
            print "Data - Model rms: {0:0.3e}".format(imdiff.std())
            axarr[row,0].imshow(imdata,interpolation='nearest',extent=ext,cmap=cmap)
            axarr[row,0].contour(imdata,extent=ext,colors='k',origin='image',levels=s*mapcontours)
            axarr[row,0].set_xlim(limits[0],limits[1]); axarr[row,0].set_ylim(limits[2],limits[3])
            axarr[row,1].imshow(immodel,interpolation='nearest',extent=ext,cmap=cmap,\
                  vmin=imdata.min(),vmax=imdata.max())
            axarr[row,1].contour(immodel,extent=ext,colors='k',origin='image',levels=s*mapcontours)
            axarr[row,1].set_xlim(limits[0],limits[1]); axarr[row,1].set_ylim(limits[2],limits[3])
            axarr[row,2].imshow(imdiff,interpolation='nearest',extent=ext,cmap=cmap,\
                  vmin=imdata.min(),vmax=imdata.max())
            axarr[row,2].contour(imdiff,extent=ext,colors='k',origin='image',levels=s*rescontours)
            axarr[row,2].set_xlim(limits[0],limits[1]); axarr[row,2].set_ylim(limits[2],limits[3])
            if np.log10(s) < -6.: sig,unit = 1e9*s,'nJy'
            elif np.log10(s) < -3.: sig,unit = 1e6*s,'$\mu$Jy'
            elif np.log10(s) < 0.: sig,unit = 1e3*s,'mJy'
            else: sig,unit = s,'Jy'
            axarr[row,2].text(0.1,0.1,"1$\sigma$ = {0:.1f}{1:s}".format(sig,unit),
                  transform=axarr[row,2].transAxes,bbox=dict(fc='w'))

            # Give a zoomed-in view in the last panel
            # Create model image at higher res, remove unlensed sources
            src = [src for src in source if src.lensed]
            imemit,_ = create_modelimage(lens,src,xemit,yemit,xemit,yemit,\
                  [0,xemit.shape[1],0,xemit.shape[0]],sourcedatamap=sourcedatamap[row])

            images[row].append(imemit)
            
            xcen = center_of_mass(imemit)[1]*(xemit[0,1]-xemit[0,0]) + xemit.min()
            ycen = -center_of_mass(imemit)[0]*(xemit[0,1]-xemit[0,0]) + yemit.max()
            dx = 0.5*(xemit.max()-xemit.min())
            dy = 0.5*(yemit.max()-yemit.min())
            
            
            if logmodel: norm=SymLogNorm(0.01*imemit.max()) #imemit = np.log10(imemit); vmin = imemit.min()-2.
            else: norm=None #vmin = imemit.min()
            axarr[row,3].imshow(imemit,interpolation='nearest',\
                  extent=[xemit.min(),xemit.max(),yemit.min(),yemit.max()],cmap=cmap,norm=norm)
            
            axarr[row,3].set_xlim(xcen-dx,xcen+dx); axarr[row,3].set_ylim(ycen-dy,ycen+dy)
            
            s = imdiff.std()
            if np.log10(s) < -6.: sig,unit = 1e9*s,'nJy'
            elif np.log10(s) < -3.: sig,unit = 1e6*s,'$\mu$Jy'
            elif np.log10(s) < 0.: sig,unit = 1e3*s,'mJy'
            else: sig,unit = s,'Jy'
            
            # Label some axes and such
            axarr[row,0].set_title(plotdata[row].filename+'\nDirty Image')
            axarr[row,1].set_title('Model Dirty Image')
            axarr[row,2].set_title('Residuals - {0:.1f}{1:s} rms'.format(sig,unit))
            if logmodel: axarr[row,3].set_title('High-res Model (log-scale)')
            else: axarr[row,3].set_title('High-res Model')
            
      

      if returnimages: return f,axarr,images
      else: return f,axarr
      
