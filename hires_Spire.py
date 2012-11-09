# This  is SPIRE-specific code for HIRES

import hires as hires

import pyfits
import numpy
import glob

hires.set_FLUX_UNITS('Jy/beam')

#====================================================================
# Override hires.read_all_IN_files method
def read_all_Spire_files():
    ''' Read all SPIRE level products from the input directory
        and return a list of Sample objects
    '''
    file_list = sorted( glob.glob(hires.INFILE_PREFIX + '*.fits') )
    if len(file_list)==0: hires.log(hires.LOG_fatal, 'No input files for: ' + \
        hires.INFILE_PREFIX) # QUIT
    if (hires.DIAG_PRODUCT):
        diagHdu = pyfits.open(hires.DIAG_PRODUCT)
        diagTable = diagHdu[1]
        chanNames = diagTable.data.field('channelName')
        scanNumbers = diagTable.data.field('scanNumber')
        coffsets = diagTable.data.field('a0')
    samples = []
    scan = 0
    for file in file_list:
        hduList = pyfits.open(file)
        maskHdu = hduList['mask']
        signalHdu = hduList['signal']
        raHdu = hduList['ra']
        decHdu = hduList['dec']
        names = []
        for det in raHdu.data.names:
            if (det[:3] == hires.BAND and det[3] != 'T' and det[3] != 'R' and det[4] != 'P'):
                names.append(det)
        names.sort()
        
        crval1 = hires.get_FITS_keyword('CRVAL1')
        crval2 = hires.get_FITS_keyword('CRVAL2')
        projection = hires.Gnomonic(crval1, crval2)
        
        for det in names:
            if (hires.DIAG_PRODUCT):
                inz = (chanNames == det) & (scanNumbers == scan)
                offsettmp = coffsets[inz]
                if (len(offsettmp) != 1):
                    raise ValueError('Did not find exactly one offset for %s in scan %d'\
                        %(det,scan))
                a0 = coffsets[inz][0]
            else:
                a0 = 0
            ra = raHdu.data.field(det)
            dec = decHdu.data.field(det)
            sig = signalHdu.data.field(det) - a0 + hires.FLUX_OFFSET
            mask = maskHdu.data.field(det)
            inx = numpy.all([mask <= 1024,mask != 256],axis=0)
            #hires.log(3, 'Generating samples for detector %s',det)
            x, y = projection.np_lonlat2xy(ra[inx], dec[inx])
            # negate x because CDELT1 is negative
            # angle = 0.0 to ignore it
            ss = hires.SampleSet(-x, y, sig[inx], '1', 0.0)
            # ss = hires.SampleSet(-x, y, sig[inx], 1, 0.0)
            # if (i%3333 == 0):
            #   hires.log(3, 'Created %dth sample for detector %s, x=%f, y=%f, ra=%f, dec=%f',\
            #           i,det,x[i], y[i], ra[inx][i],dec[inx][i]) 
            samples.append(ss)
        scan += 1
    return samples



def read_all_Spire_tables():
    ''' Read SPIRE table products
        and return a list of Sample objects
    '''
    raHduList = pyfits.open('IN/m33_raTable2.fits')
    raHdu = raHduList[1]
    names = []
    for det in raHdu.data.names:
        if (det[:3] == 'PLW' and det[3] != 'T' and det[3] != 'R' and det[4] != 'P'):
            names.append(det)
    names.sort()
    decHduList = pyfits.open('IN/m33_decTable2.fits')
    signalHduList = pyfits.open('IN/m33_signalTable2.fits')
    maskHduList = pyfits.open('IN/m33_maskTable2.fits')
    
    samples = []
    offset = 0.03
    
    crval1 = hires.get_FITS_keyword('CRVAL1')
    crval2 = hires.get_FITS_keyword('CRVAL2')
    projection = hires.Gnomonic(crval1, crval2)

    for det in names:
        #ra = raHduList[1].data[det]
        ra = raHduList[1].data.field(det)[::1]
        dec = decHduList[1].data.field(det)[::1]
        sig = signalHduList[1].data.field(det)[::1] + offset
        mask = maskHduList[1].data.field(det)[::1]
        # ra = raHduList[1].data.field(det)[::17]
        # dec = decHduList[1].data.field(det)[::17]
        # sig = signalHduList[1].data.field(det)[::17]
        # mask = maskHduList[1].data.field(det)[::17]
        #inx = numpy.all([mask <= 1024,mask != 256],axis=0)
        inx = numpy.all([(mask & 64401) == 0])
        hires.log(3, 'Generating samples for detector %s',det)
        x, y = projection.np_lonlat2xy(ra[inx], dec[inx])
        # negate x because CDELT1 is negative
        # angle = 0.0 to ignore it
        ss = hires.SampleSet(-x, y, sig[inx], '1', 0.0)
        # ss = hires.SampleSet(-x, y, sig[inx], 1, 0.0)

        # if (i%3333 == 0):
        #   hires.log(3, 'Created %dth sample for detector %s, x=%f, y=%f, ra=%f, dec=%f',\
        #           i,det,x[i], y[i], ra[inx][i],dec[inx][i]) 
        
        samples.append(ss)
    return samples


#====================================================================
# Override hires.read_all_DRF_files method

def read_all_Spire_beams():
    ''' Read SPIRE beams
        and return a dictionary of response functions
    '''
    
    # note: DRF_SET_ID now set in plw.params
    # global DRF_SET_ID
    DRF_SET_ID = 'single'
    detHduList = pyfits.open(hires.DRF_PREFIX)
    extn = hires.DRF_EXTN
    drf_array = detHduList[extn].data
    naxis1 = detHduList[extn].header['NAXIS1']
    naxis2 = detHduList[extn].header['NAXIS2']
    #deg_per_pix = 1./3600.
    #deg_per_pix = 0.8/3600.
    deg_per_pix = abs(detHduList[extn].header['CDELT1'])
    radius_pix = naxis1 / 2
    radius_degrees = radius_pix * deg_per_pix       
    detectors = {}
    #print names
    #
    def xy2response_function(x, y):
        ''' interpolate in detector response array
              x, y in degrees (relative to DRF center)
        '''
        iFloat = (x + radius_degrees) / deg_per_pix
        jFloat = (y + radius_degrees) / deg_per_pix
        # cheap "interpolation" -- just takes nearest one
        iInt = int(iFloat+0.5)
        jInt = int(jFloat+0.5)
        if iInt<0 or iInt>=naxis1 or jInt<0 or jInt>=naxis2 :
            response = 0.0
        else:
            response = drf_array[iInt, jInt]
        return response
    #
    # for id in names:
    #     detector = hires.Detector(id, radius_degrees, xy2response_function)
    #     detectors[id] = detector

    detectors['1'] = hires.Detector(1, radius_degrees, xy2response_function)
    return detectors


hires.read_all_IN_files = read_all_Spire_files
hires.read_all_DRF_files = read_all_Spire_beams


#====================================================================
# Call hires
if __name__ == "__main__":
    hires.main()
