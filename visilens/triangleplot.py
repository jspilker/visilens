import matplotlib.pyplot as pl; pl.ioff()
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
import scipy.ndimage
import numpy as np
import re
import copy

__all__ = ['TrianglePlot_MCMC','marginalize_2d','marginalize_1d']

def TrianglePlot_MCMC(mcmcresult,plotmag=True,plotnuisance=False):
      """
      Script to plot the usual triangle degeneracies.

      Inputs:
      mcmcresult:
            The result of running the LensModelMCMC routine. We
            can figure out everything we need from there.

      plotmag:
            Whether to show the dependence of magnification on the other
            parameters (it's derived, not a fit param).

      plotnuisance:
            Whether to additionally plot various nuisance parameters, like
            the absolute location of the lens or dataset amp scalings or
            phase shifts.

      Returns:
      f,axarr:
            A matplotlib.pyplot Figure object and array of Axes objects, which
            can then be manipulated elsewhere. The goal is to send something 
            that looks pretty good, but this is useful for fine-tuning.
      """

      # List of params we'll call "nuisance"
      nuisance = ['xL','yL','ampscale_dset','astromshift_x_dset','astromshift_y_dset']
      allcols = list(mcmcresult['chains'].dtype.names)
      # Gets rid of mag for unlensed sources, which is always 1.
      allcols = [col for col in allcols if not ('mu' in col and np.allclose(mcmcresult['chains'][col],1.))]
      if not plotmag: allcols = [x for x in allcols if not 'mu' in x]
      if not plotnuisance: allcols = [x for x in allcols if not any([l in x for l in nuisance])]

      labelmap = {'xL':'$x_{L}$, arcsec','yL':'$y_{L}$, arcsec','ML':'$M_{L}$, $10^{11} M_\odot$',\
            'eL':'$e_{L}$','PAL':'$\\theta_{L}$, deg CCW from E','xoffS':'$\Delta x_{S}$, arcsec','yoffS':'$\Delta y_{S}$, arcsec',\
            'fluxS':'$F_{S}$, mJy','widthS':'$\sigma_{S}$, arcsec','majaxS':'$a_{S}$, arcsec',\
            'indexS':'$n_{S}$','axisratioS':'$b_{S}/a_{S}$','PAS':'$\phi_{S}$, deg CCW from E',\
            'shear':'$\gamma$','shearangle':'$\\theta_\gamma$',
            'mu':'$\mu_{}$','ampscale_dset':'$A_{}$',
            'astromshift_x_dset':'$\delta x_{}$, arcsec','astromshift_y_dset':'$\delta y_{}$, arcsec'}

      f,axarr = pl.subplots(len(allcols),len(allcols),figsize=(len(allcols)*3,len(allcols)*3))
      axarr[0,-1].text(-0.8,0.9,'Chain parameters:',fontsize='xx-large',transform=axarr[0,-1].transAxes)
      it = 0.
      
      for row,yax in enumerate(allcols):
            for col,xax in enumerate(allcols):
                  x,y = copy.deepcopy(mcmcresult['chains'][xax]), copy.deepcopy(mcmcresult['chains'][yax])
                  if 'ML' in xax: x /= 1e11 # to 1e11Msun from Msun
                  if 'ML' in yax: y /= 1e11
                  if 'fluxS' in xax: x *= 1e3 # to mJy from Jy
                  if 'fluxS' in yax: y *= 1e3 
                  # Figure out the axis labels...
                  if xax[-1].isdigit():
                        digit = re.search(r'\d+$',xax).group()
                        xlab = (digit+'}$').join(labelmap[xax[:-len(digit)]].split('}$'))
                  else: xlab = labelmap[xax]
                  if yax[-1].isdigit():
                        digit = re.search(r'\d+$',yax).group()
                        ylab = (digit+'}$').join(labelmap[yax[:-len(digit)]].split('}$'))
                  else: ylab = labelmap[yax]

                  # To counter outlying walkers stuck in regions of low likelihood, we use percentiles
                  # instead of std().
                  xstd = np.ediff1d(np.percentile(x,[15.87,84.13]))[0]/2.
                  ystd = np.ediff1d(np.percentile(y,[15.87,84.13]))[0]/2.
                  xmin,xmax = np.median(x)-8*xstd, np.median(x)+8*xstd
                  ymin,ymax = np.median(y)-8*ystd, np.median(y)+8*ystd
                  
                  if row > col:
                        try: marginalize_2d(x,y,axarr[row,col],\
                              extent=[xmin,xmax,ymin,ymax],bins=int(max(np.floor(x.size/1000),50)))
                        except ValueError: print xax,yax; raise ValueError("One of the columns has no dynamic range.")
                        if col > 0: pl.setp(axarr[row,col].get_yticklabels(),visible=False)
                        else: axarr[row,col].set_ylabel(ylab,fontsize='x-large')
                        if row<len(allcols)-1: pl.setp(axarr[row,col].get_xticklabels(),visible=False)
                        else: axarr[row,col].set_xlabel(xlab,fontsize='x-large')
                        axarr[row,col].xaxis.set_major_locator(MaxNLocator(5))
                        axarr[row,col].yaxis.set_major_locator(MaxNLocator(5))
                  elif row == col:
                        marginalize_1d(x,axarr[row,col],extent=[xmin,xmax],\
                              bins=int(max(np.floor(x.size/1000),50)))
                        if row<len(allcols)-1: axarr[row,col].set_xlabel(xlab,fontsize='x-large')
                        if col<len(allcols)-1: pl.setp(axarr[row,col].get_xticklabels(),visible=False)
                        axarr[row,col].xaxis.set_major_locator(MaxNLocator(5))
                        #axarr[row,col].yaxis.set_major_locator(MaxNLocator(5))
                  else:                   
                        if not (row==0 and col==len(allcols)): axarr[row,col].set_axis_off()
            axarr[0,-1].text(-0.8,0.7-it,'{0:10s} = {1:.3f} $\pm$ {2:.3f}'.format(ylab,np.median(y),ystd),\
                  fontsize='xx-large',transform=axarr[0,-1].transAxes)
            it += 0.2

      axarr[0,-1].text(-0.8,0.7-it,'DIC = {0:.0f}'.format(mcmcresult['best-fit']['DIC']),fontsize='xx-large',\
            transform=axarr[0,-1].transAxes)
      
      f.subplots_adjust(hspace=0,wspace=0)

      return f,axarr

def marginalize_2d(x,y,axobj,*args,**kwargs):
      """
      Routine to plot 2D confidence intervals between two parameters given arrays
      of MCMC samples.

      Inputs:
      x,y:
            Arrays of MCMC chain values.

      axobj:
            A matplotlib Axes object on which to plot.

      extent:
            List of [xmin,xmax,ymin,ymax] values to be used as plot axis limits

      bins: 
            Number of bins to put the chains into.
      
      levs:
            Contour levels, in sigma.
      """

      # Get values of various possible kwargs
      bins = int(kwargs.pop('bins',50))
      levs = kwargs.pop('levs',[1.,2.,3.])
      extent = kwargs.pop('extent',[x.min(),x.max(),y.min(),y.max()])
      cmap = kwargs.pop('cmap','Greys')

      cmap = cm.get_cmap(cmap.capitalize())
      cmap = cmap(np.linspace(0,1,np.asarray(levs).size))
      #cmap._init()
      #cmap._lut[:-3,:-1] = 0.
      #cmap._lut[:-3,-1] = np.linspace(1,0,cmap.N)
      #colorlevs = ([200./256]*3,[80./256]*3,[12./256]*3)
      
      Xbins = np.linspace(extent[0],extent[1],bins+1)
      Ybins = np.linspace(extent[2],extent[3],bins+1)

      # Bin up the samples. Will fail if x or y has no dynamic range
      try:
            H,X,Y = np.histogram2d(x.flatten(),y.flatten(),bins=(Xbins,Ybins))
      except ValueError: return ValueError("One of your columns has no dynamic range... check it.")

      # Generate contour levels, sort probabilities from most to least likely
      V = 1.0 - np.exp(-0.5*np.asarray(levs)**2.)
      H = scipy.ndimage.filters.gaussian_filter(H,np.log10(x.size))
      Hflat = H.flatten()
      inds = np.argsort(Hflat)[::-1]
      Hflat = Hflat[inds]
      sm = np.cumsum(Hflat)
      sm /= sm[-1]

      # Find the probability levels that encompass each sigma's worth of likelihood
      for i,v0 in enumerate(V):
            try: V[i] = Hflat[sm <= v0][-1]
            except: V[i] = Hflat[0]

      V = V[::-1]
      clevs = np.append(V,Hflat.max())
      X1, Y1 = 0.5*(X[1:] + X[:-1]), 0.5*(Y[1:]+Y[:-1])

      if kwargs.get('plotfilled',True): axobj.contourf(X1,Y1,H.T,clevs,colors=cmap)
      axobj.contour(X1,Y1,H.T,clevs,colors=kwargs.get('colors','k'),linewidths=kwargs.get('linewidths',1.5),\
            linestyles=kwargs.get('linestyles','solid'))
      axobj.set_xlim(extent[0],extent[1])
      axobj.set_ylim(extent[2],extent[3])


def marginalize_1d(x,axobj,*args,**kwargs):
      """
      Plot a histogram of x, with a few tweaks for corner plot pleasantry.

      Inputs:
      x:
            Array of MCMC samples to plot up.
      axobj:
            Axes object on which to plot.
      """

      bins = int(kwargs.pop('bins',50))
      extent = kwargs.pop('extent',[x.min(),x.max()])
      fillcolor = kwargs.pop('color','gray')

      #X = scipy.ndimage.filters.gaussian_filter(x,np.log10(x.size))

      axobj.hist(x,bins=bins,range=extent,histtype='stepfilled',color=fillcolor)
      axobj.yaxis.tick_right()
      pl.setp(axobj.get_yticklabels(),visible=False)
      axobj.set_xlim(extent[0],extent[1])


