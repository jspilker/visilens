import numpy as np
arcsec2rad = np.pi/(180.*3600.)
rad2arcsec =3600.*180./np.pi
deg2rad = np.pi/180.
rad2deg = 180./np.pi

def GenerateLensingGrid(visdata,xmax,emissionbox,fieldres=None,emitres=None):
      """
      Routine to generate two grids for lensing. The first will be a lower-resolution
      grid with resolution determined by fieldres and size determined
      by xmax. The second is a much higher resolution grid which will be used for
      the lensing itself, with resolution determined by emitres and size
      determined from emissionbox - i.e., emissionbox should contain the coordinates
      which conservatively encompass the real emission, so we only have to lens that part
      of the field at high resolution.

      Since we're going to be FFT'ing with these coordinates, the resolution isn't
      directly set-able. For the low-res full-field map, it instead is set to the next-higher
      power of 2 from what would be expected from having ~4 resolution elements across
      the synthesized beam.

      Inputs:
      visdata:
            A visdata object, used to determine the resolutions of the two grids (based
            on the maximum uvdistance in the dataset)
      xmax:
            Field size for the low-resolution grid in arcsec, which will extend from
            (-xmax,-xmax) to (+xmax,+xmax), e.g. (-30,-30) to (+30,+30)arcsec. Should be
            at least a bit bigger than the primary beam.
      emissionbox:
            A 1x4 list of [xmin,xmax,ymin,ymax] defining a box (in arcsec) which contains
            the source emission.  Coordinates should be given in arcsec relative to the
            pointing center.
      fieldres,emitres:
            Resolutions of the coarse, full-field and fine (lensed) field, in arcsec.
            If not given, suitable values will be calculated from the visibilities.

      Returns:
      xmapfield,ymapfield:
            2xN matrices containing x and y coordinates for the full-field, lower-resolution
            grid, in arcsec.
      xmapemission,ymapemission:
            2xN matrices containing x and y coordinates for the smaller, very high resolution
            grid, in arcsec.
      indices:
            A [4x1] array containing the indices of xmapfield,ymapfield which overlap with
            the high resolution grid.
      """

      # Factors higher-resolution than (1/2*max(uvdist)) to make the field and emission grids
      Nover_field = 4.
      Nover_emission = 8.

      # Allow multiple visdata objects to be passed, pick the highest resolution point of all
      uvmax = 0.
      try:
            for vis in visdata:
                  uvmax = max(uvmax,vis.uvdist.max())
      except TypeError:
            uvmax = visdata.uvdist.max()

      # Calculate resolutions of the grids
      if fieldres is None: fieldres = (2*Nover_field*uvmax)**-1.
      else: fieldres *= arcsec2rad
      if emitres is None: emitres  = (2*Nover_emission*uvmax)**-1.
      else: emitres *= arcsec2rad

      # Calculate the field grid size as a power of 2.
      NFFT = 2**np.ceil(np.log2(2*np.abs(xmax)*arcsec2rad/fieldres))
      #print NFFT

      # Calculate the grid coordinates for the larger field.
      fieldcoords = np.linspace(-np.abs(xmax),np.abs(xmax),NFFT)
      xmapfield,ymapfield = np.meshgrid(fieldcoords,fieldcoords)

      # Calculate the indices where the high-resolution lensing grid meets the larger field grid
      indices = np.round(np.interp(np.asarray(emissionbox),fieldcoords,np.arange(NFFT)))

      # Calculate the grid coordinates for the high-res lensing grid; the grids should meet at indices
      #Nemx = 1 + np.abs(indices[1]-indices[0])*np.ceil((fieldcoords[1]-fieldcoords[0])/(emitres*rad2arcsec))
      #Nemy = 1 + np.abs(indices[3]-indices[2])*np.ceil((fieldcoords[1]-fieldcoords[0])/(emitres*rad2arcsec))
      #xemcoords = np.linspace(fieldcoords[indices[0]],fieldcoords[indices[1]],Nemx)
      #yemcoords = np.linspace(fieldcoords[indices[2]],fieldcoords[indices[3]],Nemy)
      #xmapemission,ymapemission = np.meshgrid(xemcoords,yemcoords)

      # Below verbatim reproduces yashar's grid
      Nemx = 1 + np.abs(indices[1]-indices[0])*np.ceil((fieldcoords[1]-fieldcoords[0])/(2*emitres*rad2arcsec))
      Nemy = 1 + np.abs(indices[3]-indices[2])*np.ceil((fieldcoords[1]-fieldcoords[0])/(2*emitres*rad2arcsec))
      xemcoords = np.linspace(fieldcoords[indices[0]],fieldcoords[indices[1]],Nemx)
      yemcoords = np.linspace(fieldcoords[indices[2]],fieldcoords[indices[3]],Nemy)
      xmapemission,ymapemission = np.meshgrid(xemcoords,yemcoords)
      xmapemission -= (xmapemission[0,1]-xmapemission[0,0])
      ymapemission -= abs((ymapemission[1,0]-ymapemission[0,0]))
      xmapemission -= 4.65484e-5 # i have no idea where this miniscule offset comes from
      ymapemission -= 4.65484e-5


      return xmapfield,ymapfield,xmapemission,ymapemission,indices
