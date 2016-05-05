import numpy as np
import sys; sys.path.append('..')
import visilens as vl
import matplotlib.pyplot as pl; pl.ioff()
import matplotlib.cm as cm
from matplotlib.widgets import Slider,Button
from astropy.cosmology import WMAP9 as cosmo


xim = np.arange(-3.,3.,.02)
yim = np.arange(-3.,3.,.02)

xim,yim = np.meshgrid(xim,yim)

zLens,zSource = 0.8,5.656
xLens,yLens = 0.,0.
MLens,eLens,PALens = 2.87e11,0.5,70.
xSource,ySource,FSource,sSource,nSource,arSource,PAsource = 0.216,0.24,0.02,0.1,0.8,0.7,120.-90
shear,shearangle = 0.12, 120.


Lens = vl.SIELens(zLens,xLens,yLens,MLens,eLens,PALens)
Shear = vl.ExternalShear(shear,shearangle)
Source = vl.SersicSource(zSource,True,xSource,ySource,FSource,sSource,nSource,arSource,PAsource)
Dd = cosmo.angular_diameter_distance(zLens).value
Ds = cosmo.angular_diameter_distance(zSource).value
Dds = cosmo.angular_diameter_distance_z1z2(zLens,zSource).value

xsource,ysource = vl.LensRayTrace(xim,yim,[Lens,Shear],Dd,Ds,Dds)

imbg = vl.SourceProfile(xim,yim,Source,[Lens,Shear])
imlensed = vl.SourceProfile(xsource,ysource,Source,[Lens,Shear])
caustics = vl.CausticsSIE(Lens,Dd,Ds,Dds,Shear)

f = pl.figure(figsize=(12,6))
ax = f.add_subplot(111,aspect='equal')
pl.subplots_adjust(right=0.48,top=0.97,bottom=0.03,left=0.05)

cmbg = cm.cool
cmbg._init()
cmbg._lut[0,-1] = 0.

cmlens = cm.gist_heat_r
cmlens._init()
cmlens._lut[0,-1] = 0.

ax.imshow(imbg,cmap=cmbg,extent=[xim.min(),xim.max(),yim.min(),yim.max()],origin='lower')
ax.imshow(imlensed,cmap=cmlens,extent=[xim.min(),xim.max(),yim.min(),yim.max()],origin='lower')
mu = imlensed.sum()*(xim[0,1]-xim[0,0])**2 / Source.flux['value']
ax.text(0.9,1.05,'$\\mu$ = {0:.2f}'.format(mu),transform=ax.transAxes)

for i in range(caustics.shape[0]):
      ax.plot(caustics[i,0,:],caustics[i,1,:],'k-')

# Put in a bunch of sliders to control lensing parameters

axcolor='lightgoldenrodyellow'
ytop,ystep = 0.9,0.05
axzL = pl.axes([0.56,ytop,0.4,0.03],axisbg=axcolor)
axzS = pl.axes([0.56,ytop-ystep,0.4,0.03],axisbg=axcolor)
axxL = pl.axes([0.56,ytop-2*ystep,0.4,0.03],axisbg=axcolor)
axyL = pl.axes([0.56,ytop-3*ystep,0.4,0.03],axisbg=axcolor)
axML = pl.axes([0.56,ytop-4*ystep,0.4,0.03],axisbg=axcolor)
axeL = pl.axes([0.56,ytop-5*ystep,0.4,0.03],axisbg=axcolor)
axPAL= pl.axes([0.56,ytop-6*ystep,0.4,0.03],axisbg=axcolor)
axss = pl.axes([0.56,ytop-7*ystep,0.4,0.03],axisbg=axcolor)
axsa = pl.axes([0.56,ytop-8*ystep,0.4,0.03],axisbg=axcolor)
axxS = pl.axes([0.56,ytop-9*ystep,0.4,0.03],axisbg=axcolor)
axyS = pl.axes([0.56,ytop-10*ystep,0.4,0.03],axisbg=axcolor)
axFS = pl.axes([0.56,ytop-11*ystep,0.4,0.03],axisbg=axcolor)
axwS = pl.axes([0.56,ytop-12*ystep,0.4,0.03],axisbg=axcolor)
axnS = pl.axes([0.56,ytop-13*ystep,0.4,0.03],axisbg=axcolor)
axarS = pl.axes([0.56,ytop-14*ystep,0.4,0.03],axisbg=axcolor)
axPAS = pl.axes([0.56,ytop-15*ystep,0.4,0.03],axisbg=axcolor)

slzL = Slider(axzL,"z$_{Lens}$",0.01,3.0,valinit=zLens)
slzS = Slider(axzS,"z$_{Source}$",slzL.val,10.0,valinit=zSource)
slxL = Slider(axxL,"x$_{Lens}$",-1.5,1.5,valinit=xLens)
slyL = Slider(axyL,"y$_{Lens}$",-1.5,1.5,valinit=yLens)
slML = Slider(axML,"log(M$_{Lens}$)",10.,12.5,valinit=np.log10(MLens))
sleL = Slider(axeL,"e$_{Lens}$",0.,1.,valinit=eLens)
slPAL= Slider(axPAL,"PA$_{Lens}$",0.,180.,valinit=PALens)
slss = Slider(axss,"Shear",0.,1.,valinit=Shear.shear['value'])
slsa = Slider(axsa,"Angle",0.,180.,valinit=Shear.shearangle['value'])
slxs = Slider(axxS,"x$_{Source}$",-1.5,1.5,valinit=xSource)
slys = Slider(axyS,"y$_{Source}$",-1.5,1.5,valinit=ySource)
slFs = Slider(axFS,"S$_{Source}$",0.01,1.,valinit=FSource)
slws = Slider(axwS,"$r_{Source}$",0.001,0.3,valinit=sSource)
slns = Slider(axnS,"n$_{Source}$",0.3,3.,valinit=nSource)
slars = Slider(axarS,"AR$_{Source}$",0.1,1.,valinit=arSource)
slPAs = Slider(axPAS,"PA$_{Source}$",-180,180,valinit=PAsource)

def update(val):
      zL,zS = slzL.val,slzS.val
      xL,yL = slxL.val, slyL.val
      ML,eL,PAL = slML.val,sleL.val,slPAL.val
      sh,sha = slss.val,slsa.val
      xs,ys = slxs.val, slys.val
      Fs,ws = slFs.val, slws.val
      ns,ars,pas = slns.val,slars.val,slPAs.val
      newDd = cosmo.angular_diameter_distance(zL).value
      newDs = cosmo.angular_diameter_distance(zS).value
      newDds= cosmo.angular_diameter_distance_z1z2(zL,zS).value
      newLens = vl.SIELens(zLens,xL,yL,10**ML,eL,PAL)
      newShear = vl.ExternalShear(sh,sha)
      newSource = vl.SersicSource(zS,True,xs,ys,Fs,ws,ns,ars,pas)
      xs,ys = vl.LensRayTrace(xim,yim,[newLens,newShear],newDd,newDs,newDds)
      imbg = vl.SourceProfile(xim,yim,newSource,[newLens,newShear])
      imlensed = vl.SourceProfile(xs,ys,newSource,[newLens,newShear])
      caustics = vl.CausticsSIE(newLens,newDd,newDs,newDds,newShear)

      ax.cla()

      ax.imshow(imbg,cmap=cmbg,extent=[xim.min(),xim.max(),yim.min(),yim.max()],origin='lower')
      ax.imshow(imlensed,cmap=cmlens,extent=[xim.min(),xim.max(),yim.min(),yim.max()],origin='lower')
      mu = imlensed.sum()*(xim[0,1]-xim[0,0])**2 / newSource.flux['value']
      ax.text(0.9,1.05,'$\\mu$ = {0:.2f}'.format(mu),transform=ax.transAxes)

      for i in range(caustics.shape[0]):
            ax.plot(caustics[i,0,:],caustics[i,1,:],'k-')

      f.canvas.draw_idle()

slzL.on_changed(update); slzS.on_changed(update)
slxL.on_changed(update); slyL.on_changed(update)
slML.on_changed(update); sleL.on_changed(update); slPAL.on_changed(update)
slss.on_changed(update); slsa.on_changed(update)
slxs.on_changed(update); slys.on_changed(update)
slFs.on_changed(update); slws.on_changed(update)
slns.on_changed(update); slars.on_changed(update); slPAs.on_changed(update)

resetax = pl.axes([0.56,ytop-16*ystep,0.08,0.04])
resbutton = Button(resetax,'Reset',color=axcolor,hovercolor='0.975')
def reset(event):
      slzL.reset(); slzS.reset()
      slxL.reset(); slyL.reset()
      slML.reset(); sleL.reset(); slPAL.reset()
      slss.reset(); slsa.reset()
      slxs.reset(); slys.reset()
      slFs.reset(); slws.reset()
      slns.reset(); slars.reset(); slPAs.reset()
resbutton.on_clicked(reset)

pl.show()
