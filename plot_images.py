import numpy as np
from scipy.ndimage.measurements import center_of_mass
from astropy.stats import sigma_clip
import matplotlib.pyplot as pl
import matplotlib.cm as cm
from matplotlib.colors import SymLogNorm
from Model_objs import *
from Data_objs import *
from uvimage import uvimageslow
from calc_likelihood import create_modelimage,fft_interpolate
from modelcal import model_cal
from GenerateLensingGrid import GenerateLensingGrid
from utils import *
import copy

def plot_images(data,mcmcresult,returnimages=False,plotcombined=False,plotall=False,
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

      # Set up to create the model image. We'll assume the best-fit values are all the medians.
      lens,source = copy.deepcopy(mcmcresult['lens_p0']), copy.deepcopy(mcmcresult['source_p0'])
      for i,ilens in enumerate(lens):
            if ilens.__class__.__name__ == 'SIELens':
                  for key in ['x','y','M','e','PA']:
                        if not vars(ilens)[key]['fixed']:
                              ilens.__dict__[key]['value'] = np.median(c[key+'L'+str(i)])
      # now do the source(s)
      for i,src in enumerate(source): # Source is a list of source objects
            if src.__class__.__name__ == 'GaussSource':
                  for key in ['xoff','yoff','flux','width']:
                        if not vars(src)[key]['fixed']:
                              src.__dict__[key]['value'] = np.median(c[key+'S'+str(i)])
            elif src.__class__.__name__ == 'SersicSource':
                  for key in ['xoff','yoff','flux','reff','index','axisratio','PA']:
                        if not vars(src)[key]['fixed']:
                              src.__dict__[key]['value'] = np.median(c[key+'S'+str(i)])
            elif src.__class__.__name__ == 'PointSource':
                  for key in ['xoff','yoff','flux']:
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

      
      if plotall:
            f,axarr = pl.subplots(len(datasets)+1,4,figsize=(12,3.5*(len(datasets)+1)))
            axarr = np.atleast_2d(axarr)
            images = [[] for _ in range(len(datasets)+1)]
      elif plotcombined:
            f,axarr = pl.subplots(1,4,figsize=(12,3))
            axarr = np.atleast_2d(axarr)
            images = [[]]
      else:
            f,axarr = pl.subplots(len(datasets),4,figsize=(12,3.5*len(datasets)))
            axarr = np.atleast_2d(axarr)
            images = [[] for _ in range(len(datasets))] # effing mutable lists.
            
      plotdata,plotinterp = [],[]
      
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
                  images[row].append(imdata); images[row].append(immodel)#; images[i].append(immap)
            
            # Plot everything up
            ext = [-imsize*pixsize/2.,imsize*pixsize/2.,-imsize*pixsize/2.,imsize*pixsize/2.]
            # Figure out what to use as the noise level; sum of weights if no user-supplied value
            if level is None: s = ((plotdata[row].sigma**-2.).sum())**-0.5
            else:
                  try:
                        s = [e for e in level][i]
                  except TypeError:
                        s = float(level)
            
            print "Data - Model rms: ",imdiff.std()
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
            #axarr[i,3].imshow(immap,interpolation='nearest',\
            #      extent=[xmap.min(),xmap.max(),xmap.max(),xmap.min()],cmap=cmap)

            # Give a zoomed-in view in the last panel
            # Create model image at higher res, remove unlensed sources
            src = [src for src in source if src.lensed]
            imemit,_ = create_modelimage(lens,src,shear,xemit,yemit,xemit,yemit,\
                  [0,xemit.shape[1],0,xemit.shape[0]],sourcedatamap)

            images[row].append(imemit)
            
            xcen = center_of_mass(imemit)[1]*(xemit[0,1]-xemit[0,0]) + xemit.min()
            ycen = center_of_mass(imemit)[0]*(xemit[0,1]-xemit[0,0]) + yemit.min()
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
      
