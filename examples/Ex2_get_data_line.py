"""
Example 2, to be run from within CASA. This script shows a bit
more complicated example for getting data out of CASA and into
a useful format. In this case, we observed an object with ATCA
looking for CO(2-1) emission at z=5.7. The line is faint, so
there were many hours of observations, spread over several days
and several different array configurations. For this example, I 
output one file per track, containing all the data from that 
day's observations (though you could just as easily combine all 
the data together and then run this).

This time, we'll average over 200km/s, centered at the rest
frequency of the CO line. 
"""

import numpy as np
import os
c = 299792458.0 # in m/s

width = '200km/s'
center = '0km/s'
restfreq = '34.64132GHz' # CO(2-1) obs frequency
outfolder = 'linedata'

# All the files where the data are. Like I say, this was a lot of observing.
vis = ['100412/spt0346_100412.cal','100812/spt0346_100812.cal','100912/spt0346_100912.cal', # H214
'102313/spt0346_102313.cal','102413/spt0346_102413.cal','102513/spt0346_102513.cal','102613/spt0346_102613.cal', # 6KM
'102713/spt0346_102713.cal','102813/spt0346_102813.cal','102913/spt0346_102913.cal','110813/spt0346_110813.cal', # 6KM
'011714/spt0346_011714.cal','011814/spt0346_011814.cal', # 1.5KM
'051014_0346/spt0346_051014.cal','051114_0346/spt0346_051114.cal', # 1.5KM
'051514/spt0346_051514.cal','051614/spt0346_051614.cal','051714/spt0346_051714.cal'] # 1.5KM

# We'll average down in time, but don't want to reduce amplitude of sources far from phase center;
# this 'magic' factor follows Cotton2009 (effects of baseline-dependent time averaging
# of uv data). It assumes a 1% loss in amplitude due to smearing for a source at 1/4
# of a PB FWHM (~100"/4 for 34.6GHz and 22m dishes), and is the max baseline over which we can time
# average in m. This value is specific to the ATCA 22m dishes and this frequency, but can be solved for with
# maxuvw = D / (1.2*loc/PBfwhm) * 0.244, where 0.244 solves 1/1.01 = sinc(x), loc = 0.25*PBfwhm
maxuvw = 18.0

splitvislist = []
for i,date in enumerate(vis):
      outms = date.split('/')[1][:-4]+'_'+center[:-4]+'_dv'+width[:-4]+'.cal'
      os.system('rm -rf '+outms)
      os.system('rm -rf '+outfolder+outms)
      
      # Average the data in both time and frequency, according to line with requested at top of script.
      mstransform(vis=date,outputvis=outms,datacolumn='data',regridms=True,restfreq=restfreq,
            mode='velocity',nchan=1,width=width,start=center,
            timeaverage=True,timebin='120s',maxuvwdistance=maxuvw,timespan='scan') # average in time, unless uv coords more than maxuvw apart
      splitvislist.append(folder+outms)
      
      # Similar to continuum script, get all the stuff we need
      ms.open(outms,nomodify=True)
      visdata = ms.getdata(['uvw','antenna1','antenna2','data','sigma','data_desc_id'])
      visdata['data'] = np.squeeze(visdata['data'])
      ms.close()
      
      # Get the frequency of this channel, for primary beam in arcsec below
      tb.open(outms+'/SPECTRAL_WINDOW')
      freqs = np.squeeze(tb.getcol('CHAN_FREQ'))
      tb.close()
      
      # Get the primary beam size
      tb.open(outms+'/ANTENNA')
      diam = np.squeeze(tb.getcol('DISH_DIAMETER'))[0] # assumes all dishes are equal size
      tb.close()
      PBfwhm = 1.2*(c/np.mean(freqs))/diam * (3600*180/np.pi) # in arcsec
      
      # Current data+sigma has two polarizations/correlations, average them.
      visdata['data'] = np.average(visdata['data'],weights=(visdata['sigma']**-2.),axis=0)
      visdata['sigma'] = np.sum((visdata['sigma']**-2.),axis=0)**-0.5
      
      # Convert uvw coords from m to wavelengths
      visdata['uvw'] *= freqs/c
      
      # Now we re-scale the noise, based on differencing successive visibilities on the same baseline
      facs = [] # for calculating a single rescale factor
      for ant1 in np.unique(visdata['antenna1']):
            for ant2 in np.unique(visdata['antenna2']):
                  if ant1 < ant2:
                        thisbase = (visdata['antenna1']==ant1) & (visdata['antenna2']==ant2)
                        reals = visdata['data'].real[:,thisbase]
                        imags = visdata['data'].imag[:,thisbase]
                        sigs = visdata['sigma'][:,thisbase]
                        diffrs = reals - np.roll(reals,-1); diffis = imags - np.roll(imags,-1)
                        std = np.mean([diffrs.std(),diffis.std()])
                        facs.append(std/sigs.mean()/np.sqrt(2)) # for calculating a single rescaling
      
      facs = np.asarray(facs)
      visdata['sigma'] *= facs.mean()
      print outms, '| mean rescaling factor: ',facs.mean(), '| rms/beam (mJy): ',1000*((visdata['sigma']**-2).sum())**-0.5
      
      # Write the new weights back to the ms file, except for having averaged polarizations together
      ms.open(outms,nomodify=False)
      replace = ms.getdata(['sigma','weight'])
      replace['sigma'] *= facs.mean()
      replace['weight'] = replace['sigma']**-2.
      ms.putdata(replace)
      ms.close()
      
      # Now we assemble the data to write it to the format that the visilens code expects.
      allarr = np.vstack((visdata['uvw'][0,:],visdata['uvw'][1,:],visdata['data'].real,visdata['data'].imag,\
                          visdata['sigma'],visdata['antenna1'],visdata['antenna2']))
  
      # Write out the data
      outfname = 'linedata/'+date.split('/')[1][:-4]+'_'+center[:-4]+'_dv'+width[:-4]+'.bin'
      with open(outfname,'wb') as f:
            allarr.tofile(f)
            f.write(PBfwhm)
      os.system('mv '+outms+' '+folder)
      #os.system('rm -rf '+outms)


# The following are optional steps, if you want to put all the data together
# to play around with later.
# visbase = folder+'spt'+source+'_'+center[:-4]+'_dv'+width[:-4]
# concat(splitvislist,visbase+'.ms')
