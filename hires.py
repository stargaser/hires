# hires" python program to produce high resolution images
# creates image using "maximum correlation method" (smoothest image that is
# consistent with the input data and the detector response functions)
# ==> Must override the "read_one_IN_file" method for reading input data files

PROGRAM = 'HIRES'
VERSION = 'v1_11'

#====================================================================
# imports
#====================================================================
import sys, os, exceptions, glob, time, logging
import numpy as np
import pyfits
from math import radians, degrees, cos, sin, atan2, sqrt, pow
import itertools as iter

#====================================================================
# log message to print and/or file, and maybe exit
#====================================================================

# log levels:
LOG_debug   = 0 # debug message
LOG_extra   = 1 # extra message
LOG_info    = 2 # standard message
LOG_step    = 3 # time tagged message (e.g. start, completed, end)
LOG_warning = 4 # warning (something odd, but not really an error)
LOG_error   = 5 # error (something wrong, but may wish to continue)
LOG_fatal   = 6 # fatal (impossible to continue)

# which level to do what:
LOG_FILE_LEVEL  = 1 # put message in log file only if at least this level
LOG_PRINT_LEVEL = 0 # print message only if at least this level
LOG_QUIT_LEVEL  = 6 # quit if message is at least this level

def log(level, message, *values):
    '''
    print/log a message
    '''
    if len(values)>0: message = message % values
    if   level == LOG_debug: prelude = 'DEBUG: '
    elif level == LOG_extra: prelude = '       ... '
    elif level == LOG_info : prelude = '       '
    elif level == LOG_step : prelude = '%5.2f' % time.clock() + ' '
    elif level == LOG_warning: prelude = '** WARNING: '
    elif level == LOG_error: prelude = '**** ERROR: '
    elif level == LOG_fatal: prelude = '**** FATAL: '
    if level >= LOG_FILE_LEVEL and LOG_FILE_HANDLE is not None:
        LOG_FILE_HANDLE.write(prelude+message +'\n')
    if level >= LOG_PRINT_LEVEL:
        if level >= LOG_warning: print ' '
        print prelude+message
        if level >= LOG_warning: print ' '
    if level >= LOG_QUIT_LEVEL:
        log(LOG_step, 'End PROCESSING *** EARLY TERMINATION ***')
        exit(-1)

#====================================================================
class SampleSet:
#====================================================================
    ''' each instance contains arrays with values for the samples '''
    def __init__(self, x, y, flux, det_id, angle):
        self.x      = x      # array - delta x from center (float, degrees)
        self.y      = y      # array - delta y from center (float, degrees)
        self.flux   = flux   # array - flux (float)
        self.neg_inx = self.flux < MIN_SAMPLE_FLUX
        self.flux[self.neg_inx] = MIN_SAMPLE_FLUX
        self.det_id = det_id # array or constant - detector ID (string)
        self.angle  = angle  # array or constant - scan angle (float, degrees east of north)
                             # - use None for hires to estimated from dx,dy
                             # - use 0.0 if angle not important

#====================================================================
class FootprintSet:
#====================================================================
    def __init__(self, flux_array, resp_array, \
                  j0_im, j1_im, i0_im, i1_im, \
                  j0_ft, j1_ft, i0_ft, i1_ft ):
        self.flux_array = flux_array
        self.resp_array = resp_array
        self.j0_im = j0_im
        self.j1_im = j1_im
        self.i0_im = i0_im
        self.i1_im = i1_im
        self.j0_ft = j0_ft
        self.j1_ft = j1_ft
        self.i0_ft = i0_ft
        self.i1_ft = i1_ft

#====================================================================
class Detector:
#====================================================================
    ''' one instance for each detector, generally one DRF file '''
    all = {}
    def __init__(self, id_passed, radius_degrees, response_function):
        id   = str(id_passed)
        self.id   = id # detector ID (string)
        self.radius_degrees = radius_degrees
        self.response_function = response_function
        Detector.all[id] = self

        log(LOG_extra,  'Detector defined, id=%s', id)

all_response_arrays = {} # indexed by (detID, iFracID, jFracID, angleID)

#------------------------------------------------------------------------
def get_response_array(detector_id, iFloat, jFloat, angle):
    ''' see if appropriate footprint array has already been generated
        If so, return it, otherwise call the method to generate it
    '''
    # compose key (to use previously computed footprint array)
    angle_round = round( angle / ANGLE_TOLERANCE )
    angleID = int( angle_round )
    angle = angle_round * ANGLE_TOLERANCE
    iFrac = iFloat % 1.0 # range:  0.0 <= frac < 1.0
    jFrac = jFloat % 1.0
    iFracID = int( iFrac * FOOTPRINTS_PER_PIX ) # range: 0 <= ID < FootPerPix
    jFracID = int( jFrac * FOOTPRINTS_PER_PIX )
    footArr_key =  ( detector_id, angleID, iFracID, jFracID )
    if footArr_key in all_response_arrays:
        footArr = all_response_arrays[footArr_key]
    else: 
        delta = 1.0 / float(FOOTPRINTS_PER_PIX)
        zero = (delta / 2.0 ) - 0.5
        iFracMod = zero + iFracID * delta # rounded for all within tolerance
        jFracMod = zero + jFracID * delta
        footArr = generate_response_array(detector_id, iFracMod, jFracMod, angle)
        all_response_arrays[footArr_key] = footArr

    return footArr

#------------------------------------------------------------------------
def generate_response_array(detector_id, iFrac, jFrac, angle):
    ''' generate full footprint array for given detector at
        given angle and centered at given x,y offset from pixel center
     detector_id
     angle (degrees) 
     iFrac, jFrac offset (frac of pixel) from center of pixel -.5 <= Frac < 0.5
    '''
    detector = Detector.all[detector_id]
    radius_degrees = detector.radius_degrees
    radius_pix = int( radius_degrees / DEG_PER_PIX )
    response_func = detector.response_function
    duFrac = iFrac * DEG_PER_PIX
    dvFrac = jFrac * DEG_PER_PIX
    duArray , dvArray =  generate_response_array_coords(radius_pix, angle, duFrac, dvFrac)
    response_array = fill_in_response_array(duArray, dvArray, response_func)

    log(LOG_debug, "full array created: %s %.2f %.2f %.2f", detector_id, iFrac,jFrac, angle)

    return response_array

#------------------------------------------------------------------------
def generate_response_array_coords(radius_pix, angle, duFrac, dvFrac):
    ''' generate du,dv coordinate arrays of footprint
        returns: duArray, dvArray two np float arrays with coords in response func
    '''
    u0 = -radius_pix; u1 = radius_pix; v0 = -radius_pix; v1 = radius_pix
    ijNPIX = radius_pix + radius_pix + 1
    duArray = np.empty( (ijNPIX, ijNPIX), dtype=np.float32 )
    dvArray = np.empty( (ijNPIX, ijNPIX), dtype=np.float32 )
    radius_degrees = radius_pix * DEG_PER_PIX # (now rounded to nearest pixel)

    # create the response array if angle is non-zero
    if angle != 0.0:
        cos_angle = cos(radians(angle))
        sin_angle = sin(radians(angle))
        dx = -radius_degrees
        for i in range(ijNPIX):
            dy = -radius_degrees
            for j in range(ijNPIX):
                du =  dx * cos_angle - dy * sin_angle - duFrac
                dv =  dy * cos_angle + dx * sin_angle - dvFrac
                duArray[i, j] = du
                dvArray[i, j] = dv
                dy += DEG_PER_PIX
            dx += DEG_PER_PIX

    # create the response array if angle is zero (usually because no angle specified)
    else:
        dx = -radius_degrees
        for i in range(ijNPIX):
            dy = -radius_degrees
            for j in range(ijNPIX):
                du =  dx - duFrac
                dv =  dy - dvFrac
                duArray[i, j] = du
                dvArray[i, j] = dv
                dy += DEG_PER_PIX
            dx += DEG_PER_PIX

    return (duArray , dvArray)


#------------------------------------------------------------------------
def fill_in_response_array(duArray, dvArray, response_func):
    ''' generate footprint response array, given u & v coordinate arrays
        input: duArray, dvArray: two np float arrays with coords in response func
        returns: response_array, an np array of floats, same dimensions as input arrays
           normalized to a sum of 1.0
    '''
    iNPIX, jNPIX = duArray.shape # iNpix should equal jNPIX
    response_array = np.empty( (jNPIX, iNPIX), dtype=np.float32 )
    for i in range(iNPIX):
        for j in range(jNPIX):
            response_here = response_func(duArray[j,i], dvArray[j,i])
            response_array[j, i] = response_here
    # normalize to sum of 1.0 and return
    sum = response_array.sum()
    normalized_response_array = response_array / sum # normalize
    return normalized_response_array


#====================================================================
# Initialize parameters
#====================================================================
FITS_KEYWORDS = [] 
ITER_LIST = {}
OUTFILE_TYPES = ('flux', )
STARTING_IMAGE = 'flat'
FOOTPRINTS_PER_PIX = 1
MIN_SAMPLE_FLUX = np.finfo('d').min # lowest possible value (~ -1.8e308)
BEAM_STARTING_IMAGE = 'flat'
BEAM_SPIKE_N = 5
BEAM_SPIKE_HEIGHT = 10.0
BOOST_MAX_ITER = 0
FLUX_UNITS = '??'
LOG_FILE_HANDLE = None
DIAG_PRODUCT = None
FLUX_OFFSET = 0.0

#====================================================================
def get_paramaters(args):
#====================================================================
    ''' Process user parameters 
        Everything put into global variables (ALL CAPS).
        Global variables should generally only be set in this method.
    '''
    global INFILE_PREFIX # directory and prefix of files containing samples
    global OUTFILE_PREFIX # directory and prefix for output filenames
    global OUTFILE_TYPES # 'flux' 'cov' beam' cfv
    global DRF_PREFIX # directory and prefix of files containing response func images
    global NPIXi, NPIXj # image size = NAXIS1, NAXIS2 (pixels)
    global DEG_PER_PIX # image scale (CDELT) degrees per pixel
    global CRVAL1, CRVAL2 # coordinates of center image --- lon, lat in degrees
    global CTYPE1, CTYPE2 # coordinates type e.g. RA---TAN, DEC--TAN
    global ITER_MAX # max iteration nmber to compute
    global ITER_LIST # which iterations to write output files for
    global MIN_SAMPLE_FLUX # force all sample fluxes to at least this value
    global BOOST_MAX_ITER # max iteration do accelerated correction
    global BOOST_TYPE # type of accelerated correction to do
    global BOOST_FUNC # function to use for accelerated correction
    global STARTING_IMAGE # user supplied starting image or 'flat'
    global ANGLE_TOLERANCE # footprints re-used if within this tollerance (degrees)
    global FOOTPRINTS_PER_PIX # generate footprints at 1/N pixel accuracy
    global BEAM_SPIKE_N # NxN spikes in initial beam images
    global BEAM_SPIKE_HEIGHT # height of initial spikes for beam images
    global BEAM_STARTING_IMAGE # user supplied BEAM starting image
    global FLUX_UNITS # for FITS keyword BUNIT in flux images
    global FITS_KEYWORDS # FITS keywords to be added to output files
    global LOG_FILE_HANDLE # log file
    global BAND # string for band or channel
    global DRF_EXTN # FITS extension for response function images
    global DIAG_PRODUCT # path to SPIRE Destriper diagnostic product (input)
    global FLUX_OFFSET # offset to add to SPIRE Level 1 timelines
    
    INFILE_PREFIX  = args[1]
    OUTFILE_PREFIX = args[2]
    param_files    = ( args[3:] )

    log_filename = 'logfile.log' # default log file
    
    for filename in ( param_files ):
        for line in open(filename):
            par_and_comment = line.strip().split('#')
            par = par_and_comment[0]
            if len(par)==0: continue # just comments on this line
            words = par.split()
            if len(words)<2: log(LOG_fatal, \
                     'parameter: %s has no value in file: %s', par, filename)
            name = words[0]
            val = words[1]
            if   name == 'SIZE_NPIX':
                NPIXi = int(val)
                if len(words)>=3: NPIXj = int(words[2])
                else: NPIXj = NPIXi
            elif name == 'ARCSEC_PER_PIX': arcsec_per_pix = float(val)
            elif name == 'CRVAL1': CRVAL1 = float(val)
            elif name == 'CRVAL2': CRVAL2 = float(val)
            elif name == 'CTYPE1': CTYPE1 = val
            elif name == 'CTYPE2': CTYPE2 = val
            elif name == 'OUTFILE_TYPES': OUTFILE_TYPES = ( words[1:] )
            elif name == 'ANGLE_TOLERANCE': ANGLE_TOLERANCE = float(val)
            elif name == 'FOOTPRINTS_PER_PIX': FOOTPRINTS_PER_PIX = int(val)
            elif name == 'DRF_PREFIX': DRF_PREFIX = val
            elif name == 'STARTING_IMAGE': STARTING_IMAGE = val
            elif name == 'MIN_SAMPLE_FLUX': MIN_SAMPLE_FLUX = float(val)
            elif name == 'BEAM_SPIKE_N': BEAM_SPIKE_N = int(val)
            elif name == 'BEAM_SPIKE_HEIGHT': BEAM_SPIKE_HEIGHT = float(val)
            elif name == 'BEAM_STARTING_IMAGE': BEAM_STARTING_IMAGE = val
            elif name == 'FLUX_UNITS': FLUX_UNITS = val
            elif name == 'LOG_FILENAME': log_filename = val
            elif name == 'BAND': BAND = val
            elif name == 'DRF_EXTN': DRF_EXTN = int(val)
            elif name == 'FLUX_OFFSET': FLUX_OFFSET = float(val)
            elif name == 'DIAG_PRODUCT': DIAG_PRODUCT = val
            elif name == 'ITER_MAX': ITER_MAX = int(val)
            elif name == 'ITER_LIST':
                for w in words[1:]:
                  iter_str =  w.split(',')
                  for n_str in iter_str:
                     if n_str != '': ITER_LIST[int(n_str)] = 1
            elif name == 'BOOST_CORRECTION':
                BOOST_MAX_ITER = int(val)
                if len(words) >= 3: BOOST_TYPE = words[2]
                if   BOOST_TYPE == 'TIMES_2': BOOST_FUNC = lambda x: x+x-1.0
                elif BOOST_TYPE == 'TIMES_3': BOOST_FUNC = lambda x: x+x+x-2.0
                elif BOOST_TYPE == 'SQUARED': BOOST_FUNC = lambda x: x*x
                elif BOOST_TYPE == 'EXP_2.5': BOOST_FUNC = lambda x: pow(x, 2.5)
                elif BOOST_TYPE == 'CUBED':   BOOST_FUNC = lambda x: x*x*x
                else: log(LOG_fatal, 'Unknown BOOST type: %s', BOOST_TYPE)
            elif name == 'KWD':
                if len(words) >=3: comment = ' '.join(words[3:])
                else: comment = None
                kwd = val
                val = numerify(words[2])
                need_warn = True
                if   kwd == 'CRVAL1': CRVAL1 = val;
                elif kwd == 'CRVAL2': CRVAL2 = val;
                elif kwd == 'CTYPE1': CTYPE1 = val;
                elif kwd == 'CTYPE2': CTYPE2 = val;
                else:
                    FITS_KEYWORDS.append( ( kwd, val, comment ) )
                    need_warn = False
                if need_warn: log(LOG_warning, 'Using KWD before %s parameter is deprecated', kwd)
            elif name == '': pass
            else:
                print 'ERROR: Unknown parameter name:', name, 'in file:', filename
                exit(-2)

    # check for errors
    for t in OUTFILE_TYPES:
        if t not in ('flux', 'cov', 'beam', 'cfv'):
            log(LOG_fatal,'Illegal OUTFILE_TYPE: %s', t)

    # compute globals from pars
    DEG_PER_PIX = arcsec_per_pix / 3600.0
    ITER_LIST[ITER_MAX] = 1 # make sure to output files for final iteration 

    # open log file
    directory = os.path.dirname(log_filename) # create directory if needed
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)
    LOG_FILE_HANDLE = open(log_filename, 'w')

#====================================================================
def print_paramaters():
#====================================================================
    log(LOG_info, '\nInput data file options:')
    log(LOG_info, '  INFILE_PREFIX %s', INFILE_PREFIX)
    log(LOG_info, '  STARTING_IMAGE %s', STARTING_IMAGE)
    log(LOG_info, '  MIN_SAMPLE_FLUX %10.6g', MIN_SAMPLE_FLUX)

    log(LOG_info, '\nDRF (detector response files) to use:')
    log(LOG_info, '  DRF_PREFIX %s', DRF_PREFIX)

    log(LOG_info, '\nOutput image geometry:')
    log(LOG_info, '  NPIX %d %d', NPIXi, NPIXj)
    log(LOG_info, '  DEG_PER_PIX %.6f', DEG_PER_PIX)
    log(LOG_info, '  CRVAL1 %.5f', CRVAL1)
    log(LOG_info, '  CRVAL2 %.5f', CRVAL2)
    log(LOG_info, '  CTYPE1 %s', CTYPE1)
    log(LOG_info, '  CTYPE2 %s', CTYPE2)

    log(LOG_info, '\nOutput file options:')
    log(LOG_info, '  OUTFILE_PREFIX %s', OUTFILE_PREFIX)
    log(LOG_info, '  OUTFILE_TYPES %s', OUTFILE_TYPES)
    log(LOG_info, '  ITER_MAX %d', ITER_MAX)
    log(LOG_info, '  ITER_LIST %s', str(sorted(ITER_LIST.keys())) )
    log(LOG_info, '  FLUX_UNITS %s', FLUX_UNITS)

    if 'beam' in OUTFILE_TYPES:
        log(LOG_info, '\nBeam image file options:')
        log(LOG_info, '  BEAM_SPIKE_N %d', BEAM_SPIKE_N)
        log(LOG_info, '  BEAM_SPIKE_HEIGHT %f', BEAM_SPIKE_HEIGHT)
        log(LOG_info, '  BEAM_STARTING_IMAGE %s', BEAM_STARTING_IMAGE)

    log(LOG_info, '\nAdditional FITS keywords:')
    for k in FITS_KEYWORDS: log(LOG_info, "  KWD %s", str(k) )

    log(LOG_info, '\nAccelerated correction option:')
    if BOOST_MAX_ITER > 0: log(LOG_info, '  BOOST %s for ITER 2 to %d', BOOST_TYPE, BOOST_MAX_ITER)
    else: log(LOG_info, '  BOOST (none)')

    log(LOG_info, '\nFootprint accuracy options:')
    log(LOG_info, '  ANGLE_TOLERANCE %.2f', ANGLE_TOLERANCE)
    log(LOG_info, '  FOOTPRINTS_PER_PIX %d', FOOTPRINTS_PER_PIX)

    print " "
    log(LOG_step, 'Start PROCESSING  (' + PROGRAM  +' '+ VERSION +')')

#====================================================================
def read_all_DRF_files():
#====================================================================
    ''' Read all of the RDF (detector response function) files.
        (by calling the "read_one_DRF_file" method)
    '''
    all_detectors = {} # dictionary accessed by detector_id
    file_list = glob.glob(DRF_PREFIX + '*')
    if len(file_list)==0: log(LOG_fatal, 'No DRF files for: ' + DRF_PREFIX) # QUIT
    for file in file_list:
        id, detector = read_one_DRF_file(file)
        all_detectors[id] = detector
    log(LOG_step, 'DRF file reading complete')
    log(LOG_info, '%d DRF files read', len(all_detectors) )
    return all_detectors

#====================================================================
def read_all_IN_files():
#====================================================================
    ''' Read all of the IN (input data) files.
        (by calling the "read_one_IN_file" method)
    '''
    all_samples = []
    prev_nsamps = 0
    file_list = sorted( glob.glob(INFILE_PREFIX + '*.fits') )
    if len(file_list)==0: log(LOG_fatal, 'No input files for: ' + INFILE_PREFIX) # QUIT
    for filename in file_list:
        read_one_IN_file(filename, all_samples)
        log(LOG_extra,'read samples from %s', filename)
    log(LOG_step, 'IN data files (samples) reading complete')
    return all_samples


#====================================================================
def count_good_samples(all_samples):
#====================================================================
    ''' Scan all samples to count number of good samples '''
    x_radius = DEG_PER_PIX * float(NPIXi-1) / 2.0 # to CENTER of first,last pixel
    y_radius = DEG_PER_PIX * float(NPIXj-1) / 2.0
    total_samps = 0
    total_good  = 0
    for samps in all_samples:
        good_ones = (samps.x>-x_radius) & (samps.x<x_radius) \
                  & (samps.y>-y_radius) & (samps.y<y_radius) 
        samps.good_ones = good_ones
        n_samps = len(samps.x)
        total_samps += n_samps
        n_good = np.sum(good_ones)
        total_good  += n_good

    log(LOG_info, "image size degrees  = %.4f x %.4f", x_radius*2.0, y_radius*2.0)

    log(LOG_step, 'IN data samples scanning complete')
    log(LOG_info, '%d data samples read', total_samps)
    if total_good==0: log(LOG_fatal, 'All data samples rejected (probably out of image)' )
    log(LOG_info, '%d data samples good', total_good)
    log(LOG_info, '%d data samples rejected', total_samps - total_good)
    pct_good = float(total_good) / float(total_samps)
    if pct_good<0.50: log(LOG_warning, 'More than 50% of data samples rejected' )

    return total_good

#====================================================================
def create_all_footprints(all_samples, all_detectors):
#====================================================================
    ''' create all footprints '''

    # Create initial Footprint empty arrays
    total_good = count_good_samples(all_samples) # so can allocate arrays of exact size
    foot_flux  = np.empty(total_good, dtype=np.float32)
    foot_resp  = np.empty(total_good, dtype=np.object)
    foot_j0_im = np.empty(total_good, dtype=np.int) 
    foot_j1_im = np.empty(total_good, dtype=np.int) 
    foot_i0_im = np.empty(total_good, dtype=np.int) 
    foot_i1_im = np.empty(total_good, dtype=np.int) 
    foot_j0_ft = np.empty(total_good, dtype=np.int) 
    foot_j1_ft = np.empty(total_good, dtype=np.int) 
    foot_i0_ft = np.empty(total_good, dtype=np.int) 
    foot_i1_ft = np.empty(total_good, dtype=np.int) 

    i_offset = float(NPIXi) / 2.0
    j_offset = float(NPIXj) / 2.0

    n_fluxes_reset  = 0
    foot_N = 0
    for ss in all_samples:
        angle_ss = ss.angle
        det_id_ss = ss.det_id
        good_ss = ss.good_ones
        if type(angle_ss) is float: angle_ss = iter.repeat(angle_ss) # constant
        if type(det_id_ss) is int: det_id_ss = str(det_id_ss) # make sure it's a string
        if type(det_id_ss) is str: det_id_ss = iter.repeat(det_id_ss) # constant

        if angle_ss is None: # need to compute angle
            max_delta = DEG_PER_PIX * 16.0 # to determine non-continuous samples
            angle_ss = [None] * len(ss.x) # array of angles starts as all None
            next_x_ss = np.append(ss.x[1:], -9999.9)
            next_y_ss = np.append(ss.y[1:], -9999.9)
            n_samples = len(ss.x)
            for i, good, x, y, next_x, next_y in \
              zip(range(n_samples), ss.good_ones, ss.x, ss.y, next_x_ss, next_y_ss):
                if good == False: continue
                if i == n_samples-1: angle = angle_ss[i-1] # last sample in array
                else:
                    x_delta = next_x - x
                    y_delta = next_y - y
                    if abs(x_delta) + abs(y_delta) > max_delta: # non-continuous
                        if i>=0: angle = angle_ss[i-1] # use prev if can't compute
                    else:
                        angle = degrees( atan2(y_delta, x_delta) )
                angle_ss[i] = angle
                if angle is None: good_ss[i] = False # discard sample if can't compute angle

                # print i, x, y, next_x, next_y, angle_ss[i], good_ss[i]

        for good, x, y, flux, angle, det_id in \
          zip(good_ss, ss.x, ss.y, ss.flux, angle_ss, det_id_ss):
            if good == False: continue
            iFloat =  (x / DEG_PER_PIX ) + i_offset
            jFloat =  (y / DEG_PER_PIX ) + j_offset
            iWhole, iFrac = divmod(iFloat, 1.0)
            jWhole, jFrac = divmod(jFloat, 1.0)
            resp_array = get_response_array(str(det_id), iFrac, jFrac, angle)
            bounds = compute_footprint_bounds(resp_array, iWhole, jWhole)
            if flux < MIN_SAMPLE_FLUX:
                flux = MIN_SAMPLE_FLUX
                n_fluxes_reset += 1
            foot_flux[foot_N]   = flux
            foot_resp[foot_N]   = resp_array
            foot_j0_im[foot_N] = bounds[0]
            foot_j1_im[foot_N] = bounds[1]
            foot_i0_im[foot_N] = bounds[2]
            foot_i1_im[foot_N] = bounds[3]
            foot_j0_ft[foot_N] = bounds[4]
            foot_j1_ft[foot_N] = bounds[5]
            foot_i0_ft[foot_N] = bounds[6]
            foot_i1_ft[foot_N] = bounds[7]
            foot_N += 1
    foot_resp = foot_resp[0:foot_N]

    log(LOG_step, 'Footprint creation complete')
    log(LOG_info, '%d footprints created', foot_N )
    log(LOG_extra, '(%d full response arrays created)',  len(all_response_arrays) ) 
    log(LOG_info, '%d sample fluxes reset to minimum', n_fluxes_reset)

    return FootprintSet( foot_flux, foot_resp, \
            foot_j0_im, foot_j1_im, foot_i0_im, foot_i1_im, \
            foot_j0_ft, foot_j1_ft, foot_i0_ft, foot_i1_ft)


#====================================================================
def compute_footprint_bounds(resp_array, iFloat, jFloat):
#====================================================================
    ''' Compute boundary of response function in image (resp_array, iFloat, jFloat)
        Also compute start of response_array to use
        trims bounds if they extend out of image
        returns:
        (j0_im, j1_im, i0_im, i1_im, j0_ft, j1_ft, i0_ft, i1_ft)
    '''

    # Get size of response array
    j_size, i_size = resp_array.shape # should have j_size == i_size and ODD
    radius_pixels = j_size / 2
    j_center = int(jFloat)
    i_center = int(iFloat)
    j0_resp = 0
    j1_resp = j_size
    i0_resp = 0
    i1_resp = i_size

    # Compute nominal bounds
    j0_image = j_center - radius_pixels
    j1_image = j_center + radius_pixels + 1
    i0_image = i_center - radius_pixels     # iMin
    i1_image = i_center + radius_pixels + 1 # iMax+1

    # Trim if outside image
    if j0_image<0:
        j0_resp -= j0_image
        j0_image = 0
    elif j1_image>NPIXj:
        j1_resp -= j1_image-NPIXj
        j1_image = NPIXj
    if i0_image<0:
        i0_resp -= i0_image
        i0_image = 0
    elif i1_image>NPIXi:
        i1_resp -= i1_image-NPIXi
        i1_image = NPIXi

    return (j0_image, j1_image, i0_image, i1_image, j0_resp, j1_resp, i0_resp, i1_resp)
     


########################################################
# FITS: first pix called "1"  (center is 1.0)
#        center of center pix is 3.0
#        e.g.  if NAXIS = 5 CRPIX=3.0  pixels centered at 1.0, 2.0, 3.0, 4.0, 5.0
#               entire range goes 0.5 to 5.5
# 
# 
#       radius_pixels = float(NPIX) / 2.0
#       radius_degrees = CDELT * radius_pixels
# 
#       extended_radius_pixels = radius_pixels + 0.5
#       extended_radius_degrees = CDELT * extended_radius_pixels
#                               = CDELT * (radius_pixels + 0.5)
# 
# 
# x,y:  center pix center is 0.0
#       first pix center is -radius_degrees,  edgo of pixel at "extended_radius")
#       last  pix center is +radius_degrees
# 
# Only use samples if inside *UNEXTENDED* radius
# 
# 
# Python i,j: first pix called "0" (edge at 0.0;  center at 0.5)
#   need to be able to do a divmod on it and get init, frac
#        center pix is 2;  (center of center pix is 2.5)
#               entire float range goes 0.0 to 4.999999
#               entire int range goes 0 to 4, but slice called "0:5"
#               pixel N covered by N+0 to N+0.9999999
#               int,frac = divmod(float, 1.0)
# 
# 
# CRPIX = float(NAXIS+1) / 2.0
# 
# e.g. NAXIS = 5
#      CRPIX = 6/2 = 3.0
# 
#  i = x / CDELT + float(NAXIS)/2.0
#    = x / CDELT + (CRPIX - 0.5)
########################################################


#====================================================================
def calc_wgt_image(fp):
#====================================================================
    ''' Create an image of all footprints *1.0 '''
    wgt_image = np.empty( (NPIXj, NPIXi), dtype=np.float32)
    wgt_image.fill(0.00000001) # to avoid NaNs in results
    for resp, j0_im, j1_im, i0_im, i1_im, j0_ft, j1_ft, i0_ft, i1_ft  \
      in zip(fp.resp_array, fp.j0_im, fp.j1_im, fp.i0_im, fp.i1_im, \
                            fp.j0_ft, fp.j1_ft, fp.i0_ft, fp.i1_ft):
          wgt_image[ j0_im:j1_im, i0_im:i1_im ] \
            += resp[ j0_ft:j1_ft, i0_ft:i1_ft ]
    log(LOG_step, 'Weight array computed')
    return wgt_image

#====================================================================
def make_start_image(filename):
#====================================================================
    ''' starting image - use flat image if 'flat'
       otherwise read in starting image '''
    iter_start = 0 # will be set non-zero if in starting image selected
    if filename == 'flat': # make a flat image to start
        image = np.zeros( (NPIXj, NPIXi), dtype=np.float32 )
        background = 1.0
        image += background
    else: # read image from file
        hduList = pyfits.open(filename)
        kwd = hduList[0].header
        # check that dimensions are correct
        if (kwd['NAXIS1'] != NPIXi) or (kwd['NAXIS2'] != NPIXj):
            log(LOG_fatal, "STARTING_IMAGE %s has incorrect dimensions\n" + \
             '   Must be: %d %d ', filename, NPIXi, NPIXj)
        if abs( kwd['CRVAL1'] - CRVAL1 ) > 0.001:
            log(LOG_error, "STARTING_IMAGE %s has inconsistent CRVAL1 value(s)\n" + \
             '   Should be: %.5f', filename, CRVAL1)
        if abs( kwd['CRVAL2'] - CRVAL2 ) > 0.001:
            log(LOG_error, "STARTING_IMAGE %s has inconsistent CRVAL2 value(s)\n" + \
             '   Should be: %.5f', filename, CRVAL2)
        if (kwd['BUNIT'] != FLUX_UNITS):
            log(LOG_error, "STARTING_IMAGE %s has inconsistent BUNIT\n" + \
             "   Should be: '%s'", filename, FLUX_UNITS)
        if kwd.has_key('ITERNUM'):
            iter_start = kwd['ITERNUM']
            log(LOG_info, 'ITERNUM in %s is %d', filename, iter_start)
        else: log(LOG_warning, 'No ITERNUM keyword in START_IMAGE %s', filename)
        image = hduList[0].data
    return (image, iter_start)

#====================================================================
def calc_corr_wgt_image(fp, flux_image, iter_num, do_cfv):
#====================================================================
    ''' Create a correction image: For each footprint:
         1. Calculate "correcton" 
         2. Add into correction image: response_array * correction
    '''
    corr_wgt_image = np.zeros( (NPIXj, NPIXi), dtype=np.float32)
    if do_cfv: corr_sq_wgt_image = np.zeros((NPIXj, NPIXi),dtype=np.float32)
    else: corr_sq_wgt_image = None
    boost_mode = None
    for flux_sample, resp, j0_im, j1_im, i0_im, i1_im, j0_ft, j1_ft, i0_ft, i1_ft in \
      zip(fp.flux_array, fp.resp_array, fp.j0_im, fp.j1_im, fp.i0_im, fp.i1_im, \
                                        fp.j0_ft, fp.j1_ft, fp.i0_ft, fp.i1_ft):
        integration = flux_image[ j0_im:j1_im, i0_im:i1_im ] \
                          * resp[ j0_ft:j1_ft, i0_ft:i1_ft ]
        flux_prime = integration.sum()
        correction = (flux_sample / flux_prime)
        if iter_num != 1 and iter_num <= BOOST_MAX_ITER:
            correction = BOOST_FUNC(correction)
            boost_mode = "   (BOOSTED correction)"
        corr_wgt_image[ j0_im:j1_im, i0_im:i1_im ] \
               += resp[ j0_ft:j1_ft, i0_ft:i1_ft ] * correction

        if do_cfv:
            corr_sq = correction * correction
            corr_sq_wgt_image[ j0_im:j1_im, i0_im:i1_im ] \
                      += resp[ j0_ft:j1_ft, i0_ft:i1_ft ] * corr_sq
    log(LOG_step, 'Correction array computed, iter ' + str(iter_num) )
    if boost_mode is not None: log(LOG_info, boost_mode)
    return ( corr_wgt_image , corr_sq_wgt_image )

#====================================================================
def set_fluxes_to_sim_values(fp, sim_image):
#====================================================================
    ''' Calculate simulated fluxes; replace fp.flux value
        Estimates what flux values would generate the input image '''

    flux_array_sim = fp.flux_array # get original flux values
    i=0
    for resp, j0_im, j1_im, i0_im, i1_im, j0_ft, j1_ft, i0_ft, i1_ft in \
      zip(fp.resp_array, fp.j0_im, fp.j1_im, fp.i0_im, fp.i1_im, \
                         fp.j0_ft, fp.j1_ft, fp.i0_ft, fp.i1_ft):

        integration = sim_image[ j0_im:j1_im, i0_im:i1_im ] \
                         * resp[ j0_ft:j1_ft, i0_ft:i1_ft ]
        flux_array_sim[i] = integration.sum() # fill in simulated flux
        i += 1
    fp.flux_array = flux_array_sim # replace oroginal fluxes with sim

    log(LOG_step, 'Fluxes have been reset to simulated values')

#====================================================================
def create_spike_image(n, height):
#====================================================================
    ''' Create initial spike image
        (values are all small>0 except for nXn spikes of given height '''
    spike_image = np.zeros( (NPIXj, NPIXi), dtype=np.float32)
    spike_image += 0.000001
    iDelta = NPIXi / n
    jDelta = NPIXj / n
    for i in range(iDelta/2, NPIXi, iDelta):
        for j in range(jDelta/2, NPIXj, jDelta):
            spike_image[j,i] = height
    return spike_image

#====================================================================
def set_FLUX_UNITS(units):
#====================================================================
    ''' Set FLUX_UNITS global variable, the BUNIT keyword in output files.
        (intended  for use in "read_one_IN_file") '''
    global FLUX_UNITS
    FLUX_UNITS = units

#====================================================================
def get_FITS_keyword(keyword):
#====================================================================
    ''' Get a FITS keyword from FITS_KEYWORDS.  Return None if not there. '''
    if keyword == 'CRVAL1': return CRVAL1
    if keyword == 'CRVAL2': return CRVAL2
    if keyword == 'CTYPE1': return CTYPE1
    if keyword == 'CTYPE2': return CTYPE2
    for k in FITS_KEYWORDS:
        kwd = k[0]
        if kwd == keyword: return k[1]
    return None # not found

#====================================================================
def numerify (s):
#====================================================================
    ''' convert string to int or float if possible '''
    try: return int(s)
    except exceptions.ValueError:
        try: return float(s)
        except exceptions.ValueError: return s

#====================================================================
def write_FITS_image(image, file_type, iter=None):
#====================================================================
    ''' Writes out one FITS image of specified type.
         image = array with pixel values
         file_type = 'flux', 'cov', 'cfv', etc.
         iter = iteration number (put in FITS keyword)
         trim_padding = Should padding be removed ?
    '''
    directory = os.path.dirname(OUTFILE_PREFIX)
    if not os.path.exists(directory): os.makedirs(directory)
    filename = OUTFILE_PREFIX + '_' + file_type
    if iter is not None: filename += '_' + str(iter)
    filename += '.fits'

    hdu = pyfits.PrimaryHDU(image)
    prihdr = hdu.header

    # add keywords
    if (file_type=='flux'): prihdr.update('BUNIT', FLUX_UNITS)
    for kwd_tuple in FITS_KEYWORDS: # user specified kwds
        prihdr.update(kwd_tuple[0], kwd_tuple[1], comment=kwd_tuple[2])
    prihdr.update('CRVAL1', CRVAL1)
    prihdr.update('CRVAL2', CRVAL2)
    prihdr.update('CTYPE1', CTYPE1)
    prihdr.update('CTYPE2', CTYPE2)
    cdelt_rounded = float('%.7f' % DEG_PER_PIX)
    prihdr.update('CDELT1', -cdelt_rounded, comment='left(+) to right(-)')
    prihdr.update('CDELT2',  cdelt_rounded, comment='bottom(-) to top(+)')
    prihdr.update('CRPIX1', (NPIXi+1)/2 , comment='center pixel')
    prihdr.update('CRPIX2', (NPIXj+1)/2 , comment='center pixel')
    if   file_type=='flux': t_comment = 'HIRES flux image'
    elif file_type=='cov':  t_comment = 'HIRES coverage image'
    elif file_type=='cfv':  t_comment = 'HIRES correction factor variance image'
    else:                   t_comment = 'HIRES image'
    prihdr.update('FILETYPE', file_type, comment=t_comment)
    if iter is not None: prihdr.update('ITERNUM', iter, comment='HIRES iteration number')
    filename_only = filename.split('/')[-1] # strip off leading directory names
    prihdr.update('FILENAME', filename_only, comment='name of this file')  
    in_pref = INFILE_PREFIX.split('/')[-1] +'*'
    prihdr.update('INFILES' , in_pref  , comment='Name of input data files')  
    if DRF_PREFIX is not None:
        drf_pref = DRF_PREFIX.split('/')[-1] +'*'
        prihdr.update('DRF_IN'  , drf_pref  , comment='Name of Detector Response Files')  
    dt = time.strftime("%Y/%m/%d %H:%M:%S")
    prihdr.update('DATE'     , dt  , comment='when this file was created')  
    prihdr.update('CREATED', PROGRAM+' '+VERSION, comment='software version that created this file')

    # write file
    hdulist = pyfits.HDUList([hdu])
    if os.path.isfile(filename): os.remove(filename)
    hdulist.writeto(filename)
    log(LOG_step, 'Output file written: %s', filename)

#====================================================================
def read_one_DRF_file(filename):
#====================================================================
    ''' Read one detector response function file.
        Return the detector ID and a Detector object.
        This method may be overiden for project-specific DRF files.
    '''
    hduList = pyfits.open(filename)
    kwd = hduList[0].header
    id = kwd['DETECTOR']
    naxis1 = kwd['NAXIS1']
    naxis2 = kwd['NAXIS2']
    cdelt1 = kwd['CDELT1']
    cdelt2 = kwd['CDELT2']
    if naxis1 != naxis2: log(LOG_error, 'in ' + filename + ' NAXIS1 must equal NAXIS2')
    if naxis1 %2==0 : log(LOG_error, 'in ' + filename + ' NAXIS1 must be odd')
    if abs(cdelt1) != abs(cdelt2):
        log(LOG_error, 'in ' + filename + ' CDELT1 and CDELT2 must be same size')
    if cdelt1 >= 0: log(LOG_warning, 'in ' + filename + ' CDELT1 must be negative')
    if cdelt2 <= 0: log(LOG_warning, 'in ' + filename + ' CDELT2 must be positive')
    deg_per_pix = cdelt2

    radius_pix = naxis1 / 2
    radius_degrees = radius_pix * deg_per_pix

    drf_array = hduList[0].data

    def duv2response_function(du, dv):
        ''' interpolate in detector response array
         du, dv in degrees relative to DRF center
         '''

        iFloat = (du + radius_degrees) / deg_per_pix
        jFloat = (dv + radius_degrees) / deg_per_pix
        # cheap "interpolation" -- just takes nearest one
        iInt = int(iFloat+0.5)
        jInt = int(jFloat+0.5)
        if iInt<0 or iInt>=naxis1 or jInt<0 or jInt>=naxis2 :
            response = 0.0
        else:
            response = drf_array[jInt, iInt]
        return response

    log(LOG_step, 'detector: %d  file=%s radius= %d pixels = %f arcmin', \
      id, filename.split('/')[-1], radius_pix, radius_degrees*60 ) 
    detector = Detector(id, radius_degrees, duv2response_function)
    return ( id, detector )

#====================================================================
class Gnomonic:
#====================================================================
    ''' Compute gnomonic (tangent plane) projection of lon,lat to x,y
        lat,lon,x,y all in degrees
        This method intended to be used by overrides of "read_one_IN_file".
        Equations from: http://mathworld.wolfram.com/GnomonicProjection.html
        Example:
          gn = Gnomonic(lon_center_degrees, lat_center_degrees)
          x, y = lonlat2xy(lon_degrees, lat_degrees)
    '''
    def __init__(self, lon_degrees, lat_degrees):
        lat0 = radians(lat_degrees)
        self.sin_lat0  = sin(lat0)
        self.cos_lat0  = cos(lat0)
        self.lon0 = radians(lon_degrees)

    def np_lonlat2xy(self, lon_degrees, lat_degrees):
        ''' inputs and outputs are NumPy arrays) '''
        lat = np.radians(lat_degrees)
        lon = np.radians(lon_degrees)
        cos_dlon = np.cos(lon - self.lon0)
        sin_dlon = np.sin(lon - self.lon0)
        sin_lat  = np.sin(lat)
        cos_lat  = np.cos(lat)
        cos_c = np.radians( self.sin_lat0 * sin_lat + self.cos_lat0 * cos_lat * cos_dlon )
        x = (cos_lat * sin_dlon) / cos_c
        # y = (self.sin_lat0 * cos_lat * cos_dlon - self.cos_lat0 * sin_lat) / cos_c
        y = (self.cos_lat0 * sin_lat - self.sin_lat0 * cos_lat * cos_dlon) / cos_c
        return (x, y)

    def lonlat2xy(self, lon_degrees, lat_degrees):
        lat = radians(lat_degrees)
        lon = radians(lon_degrees)
        cos_dlon = cos(lon - self.lon0)
        sin_dlon = sin(lon - self.lon0)
        sin_lat  = sin(lat)
        cos_lat  = cos(lat)
        cos_c = radians( self.sin_lat0 * sin_lat + self.cos_lat0 * cos_lat * cos_dlon )
        x = (cos_lat * sin_dlon) / cos_c
        # y = (self.sin_lat0 * cos_lat * cos_dlon - self.cos_lat0 * sin_lat) / cos_c
        y = (self.cos_lat0 * sin_lat - self.sin_lat0 * cos_lat * cos_dlon) / cos_c
        return (x, y)

#====================================================================
def read_one_IN_file(filename, all_samples):
#====================================================================
    ''' Read one IN (input data) file, appending SampleSet object to all_samples.
        This method **must** be overiden for project-specific "IN" files.
    '''
    print 'PROGRAMMING ERROR: Must override read_one_IN_file method'
    exit(-1)

#====================================================================
# MAIN PROGRAM
#====================================================================

def main():

    #-----------------------------------------------------------
    # Initialize
    #-----------------------------------------------------------
    get_paramaters(sys.argv)
    print_paramaters()
    all_detectors = read_all_DRF_files()
    all_samples = read_all_IN_files()
    all_footprints = create_all_footprints(all_samples, all_detectors)
    del all_samples
    wgt_image = calc_wgt_image(all_footprints)
    if 'cov' in OUTFILE_TYPES: write_FITS_image( wgt_image, 'cov')

    #-----------------------------------------------------------
    # Create FLUX image(s)
    #-----------------------------------------------------------
    if 'flux' in OUTFILE_TYPES:
        flux_image, iter_start = make_start_image(STARTING_IMAGE)
        for iter in range(iter_start+1, ITER_MAX+1):
            do_cfv_image =  ('cfv' in OUTFILE_TYPES) and (iter in ITER_LIST)
            corr_wgt_image, corr_sq_wgt_image = \
              calc_corr_wgt_image(all_footprints, flux_image, iter, do_cfv_image)
            correction_image = corr_wgt_image / wgt_image
            flux_image *= correction_image
            log(LOG_extra, 'Mean flux in image = %f', flux_image.mean() )
            if iter in ITER_LIST: write_FITS_image(flux_image, 'flux', iter)
            if do_cfv_image: 
                corr_sq_image = (corr_sq_wgt_image / wgt_image) - \
                                (correction_image * correction_image)
                write_FITS_image(corr_sq_image, 'cfv', iter)

    #-----------------------------------------------------------
    # Create BEAM image(s)
    #-----------------------------------------------------------
    if 'beam' in OUTFILE_TYPES:
        spike_image = create_spike_image(BEAM_SPIKE_N, BEAM_SPIKE_HEIGHT)
        set_fluxes_to_sim_values(all_footprints, spike_image) # reset Sample.flux
        beam_image, iter_start = make_start_image(BEAM_STARTING_IMAGE)
        for iter in range(iter_start+1, ITER_MAX+1):
            corr_wgt_image, corr_sq_wgt_image = \
              calc_corr_wgt_image(all_footprints, beam_image, iter, False)
            beam_image *= corr_wgt_image / wgt_image
            if iter in ITER_LIST: write_FITS_image(beam_image, 'beam', iter)

    log(LOG_step, 'End PROCESSING')
