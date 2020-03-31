visilens
========

News / updates
--------------

I have finally, finally made the jump to python3 now that 
python2.7 has been sunset. Any and all future development will
be in a python3 environment. 

The current status is that all the syntax has been updated to
python3, but I haven't yet updated based on new external package
changes. The key one is that for now we still require a version
of `emcee` prior to 3.0 (most recent was 2.2.1) because in >3.0
many of the function calls and such have changed.

If you need a python2.7 version of this package, you can find the
final release version [here](https://github.com/jspilker/visilens/releases/tag/v1.0).
I know that version works with versions of `emcee` up to at least
2.1.0, but that package has changed a lot of function calls in versions >3.0.


About visilens
--------------

Visilens is a python module for modeling gravitational lensing
systems observed by a radio/mm interferometer like ALMA or ATCA.
Because interferometers observe the Fourier components of the
sky and not actual images, every pixel in an interferometric
image is correlated with every other pixel. If you try to make
lens models using these images, you can get models which are just
plain wrong, or at least parameter uncertainties which aren't
estimated correctly. There can be residual calibration problems
in the data which might lead you to the wrong model.

Visilens gets around this by modeling the interferometric
visibilities directly. This lets us take care of those calibration
uncertainties (from variations in the absolute flux scale, or
not knowing the antenna positions perfectly, or residual 
atmospheric effects) directly. Visilens marginalizes over these
data uncertainties to arrive at models which take full advantage
of the signal in the data, with accurate uncertainty estimates.

One of my favorite examples of how this works is shown below. We
initially got 1.5arcsec resolution imaging of one particular
source, SPT0346-52. The model kept suggesting the structure in the
upper-right part of the middle panel, and even we didn't believe
that could possibly be right - the model predicted three lensed 
images, but there's only two in the image made from the data. 
We've since gotten data down to 0.25arcsec resolution, which 
exactly confirms the original model. This tells you that there's 
information present in the visibilities which isn't obvious by eye 
in any images.

![SPT0346-52 lens models](/examples/SPT0346-52_models.png?raw=true)

How to use visilens
-------------------

There are a few demos and example usage scripts in the examples/
folder, along with the data files to reproduce the models in those
examples.

If you're new to lensing, I recommend playing around with the two
Demo scripts first, which I hope will help build intuition for how
the lens mapping / caustics change as you change the lens properties,
and how that affects the observed emission.

The first two example scripts in that folder deal with getting data
out of CASA's measurement set format and into something more useful.
There's an example for continuum data and for spectral line data.

The next two examples go through the model fitting process, and show
pretty much all the options / features of the code.

If you need help using the code, please feel free to email me.

Attribution
-----------

If you find visilens useful for your work, please cite Hezaveh et
al. (2013), and Spilker et al. (2016):

    @ARTICLE{hezaveh13a,
      author = {{Hezaveh}, Y.~D. and {Marrone}, D.~P. and {Fassnacht}, C.~D. and 
    	{Spilker}, J.~S. and {Vieira}, J.~D. and {Aguirre}, J.~E. and 
    	{Aird}, K.~A. and {Aravena}, M. and {Ashby}, M.~L.~N. and {Bayliss}, M. and 
    	{Benson}, B.~A. and {Bleem}, L.~E. and {Bothwell}, M. and {Brodwin}, M. and 
    	{Carlstrom}, J.~E. and {Chang}, C.~L. and {Chapman}, S.~C. and 
    	{Crawford}, T.~M. and {Crites}, A.~T. and {De Breuck}, C. and 
    	{de Haan}, T. and {Dobbs}, M.~A. and {Fomalont}, E.~B. and {George}, E.~M. and 
    	{Gladders}, M.~D. and {Gonzalez}, A.~H. and {Greve}, T.~R. and 
    	{Halverson}, N.~W. and {High}, F.~W. and {Holder}, G.~P. and 
    	{Holzapfel}, W.~L. and {Hoover}, S. and {Hrubes}, J.~D. and 
    	{Husband}, K. and {Hunter}, T.~R. and {Keisler}, R. and {Lee}, A.~T. and 
    	{Leitch}, E.~M. and {Lueker}, M. and {Luong-Van}, D. and {Malkan}, M. and 
    	{McIntyre}, V. and {McMahon}, J.~J. and {Mehl}, J. and {Menten}, K.~M. and 
    	{Meyer}, S.~S. and {Mocanu}, L.~M. and {Murphy}, E.~J. and {Natoli}, T. and 
    	{Padin}, S. and {Plagge}, T. and {Reichardt}, C.~L. and {Rest}, A. and 
    	{Ruel}, J. and {Ruhl}, J.~E. and {Sharon}, K. and {Schaffer}, K.~K. and 
    	{Shaw}, L. and {Shirokoff}, E. and {Stalder}, B. and {Staniszewski}, Z. and 
    	{Stark}, A.~A. and {Story}, K. and {Vanderlinde}, K. and {Wei{\ss}}, A. and 
    	{Welikala}, N. and {Williamson}, R.},
      title = "{ALMA Observations of SPT-discovered, Strongly Lensed, Dusty, Star-forming Galaxies}",
      journal = {\apj},
      archivePrefix = "arXiv",
      eprint = {1303.2722},
      keywords = {galaxies: high-redshift, galaxies: starburst, gravitational lensing: strong, techniques: interferometric},
      year = 2013,
      month = apr,
      volume = 767,
      eid = {132},
      pages = {132},
      doi = {10.1088/0004-637X/767/2/132},
      adsurl = {http://adsabs.harvard.edu/abs/2013ApJ...767..132H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
    
    @ARTICLE{spilker16a,
      author = {{Spilker}, J.~S. and {Marrone}, D.~P. and {Aravena}, M. and 
    	{B{\'e}thermin}, M. and {Bothwell}, M.~S. and {Carlstrom}, J.~E. and 
    	{Chapman}, S.~C. and {Crawford}, T.~M. and {de Breuck}, C. and 
    	{Fassnacht}, C.~D. and {Gonzalez}, A.~H. and {Greve}, T.~R. and 
    	{Hezaveh}, Y. and {Litke}, K. and {Ma}, J. and {Malkan}, M. and 
    	{Rotermund}, K.~M. and {Strandet}, M. and {Vieira}, J.~D. and 
    	{Weiss}, A. and {Welikala}, N.},
      title = "{ALMA Imaging and Gravitational Lens Models of South Pole Telescope{\mdash}Selected Dusty, Star-Forming Galaxies at High Redshifts}",
      journal = {\apj},
      archivePrefix = "arXiv",
      eprint = {1604.05723},
      keywords = {galaxies: high-redshift, galaxies: ISM, galaxies: star formation },
      year = 2016,
      month = aug,
      volume = 826,
      eid = {112},
      pages = {112},
      doi = {10.3847/0004-637X/826/2/112},
      adsurl = {http://adsabs.harvard.edu/abs/2016ApJ...826..112S},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

License
-------

visilens is free software provided under the MIT license. You can
read the legalese version in LICENSE.txt.
