import numpy as np
import os 
import glob
import time
import sys
import subprocess
import astropy.units as u
from astropy.time import Time
import sigproc as fb
import click

#########################
##  DADA HEADER CLASS  ##
#########################

class DADA_HDR:
    def __init__(self):
        self.obs_id = "unset"
        self.filename = "unset"
        self.filenum = 0
        self.file_size = 0
        self.obs_offset = 0
        self.obs_overlap = 0
        self.source_name = "unset"
        self.telescope = "DSN"
        self.receiver = "unset"
        self.nbit = 8 
        self.ndim = 2 
        self.npol = 1 
        self.nchan = 7
        self.tsamp = 0.0  # microsec
        self.bw = 0.0     # MHz
        self.utc_start = "2021-03-02-23:52:10.999999749"
        self.mjd_start = 0.0
        self.freq = 0.0 

    def ascii_hdr(self):
        outstr =  (
        "HEADER DADA\n" + 
        "HDR_VERSION 1.0\n" + 
        "HDR_SIZE 4096\n" + 
        "DADA_VERSION 1.0\n" +
        "OBS_ID %s\n" %(self.obs_id) + 
        "PRIMARY unset\n" + 
        "SECONDARY unset\n" + 
        "FILE_NAME %s\n" %(self.filename) + 
        "FILE_NUMBER %d\n" %(self.filenum) + 
        "FILE_SIZE %d\n" %(self.file_size) +  
        "OBS_OFFSET %d\n" %(self.obs_offset) + 
        "OBS_OVERLAP %s\n" %(self.obs_overlap) + 
        "SOURCE %s\n" %(self.source_name) + 
        "TELESCOPE %s\n" %(self.telescope) + 
        "INSTRUMENT DADA\n" + 
        "RECEIVER %s\n" %(self.receiver) + 
        "NBIT %d\n" %(self.nbit) + 
        "NDIM %d\n" %(self.ndim) + 
        "NPOL %d\n" %(self.npol) + 
        "NCHAN %d\n" %(self.nchan) + 
        "RESOLUTION 1\n" + 
        "DSB 1\n" + 
        "TSAMP %.8f\n" %(self.tsamp) + 
        "BW %8f\n" %(self.bw) + 
        "UTC_START %s\n" %(self.utc_start) + 
        "MJD_START %.16f\n" %(self.mjd_start) + 
        "FREQ %.8f\n" %(self.freq) + 
        "# end of header\n")
        return outstr

    def write_header(self, outfile):
        """
        Write PSRDADA header -- this consists of 
        an ASCII part then pad "\x00" until you 
        get to 4096 bytes  
        """ 
        # First we need to add the file name to header
        self.filename = outfile

        # Next we get the byte size of the header
        hdr_str = self.ascii_hdr()
        hdr_bytes = len(hdr_str.encode('utf-8'))
        
        # Now get number of bytes to fill to 4096
        nbyte_fill = 4096 - hdr_bytes

        # We will pad the header using unsigned int 0s
        fill_bytes = np.zeros(nbyte_fill, dtype='uint8') 

        # Open the file and write ascii header
        with open(outfile, 'w') as fout:
            fout.write(hdr_str)

        # Now append the file with padded 0s
        with open(outfile, 'ab') as fout:
            fout.write(fill_bytes)

        return


###############################
##  READ DSN CS DATA FILES   ##
###############################

def get_cs_info(infile):
    hd, hsize, err = fb.read_header(infile, 4, fb.fmtdict)
    if err:
        print("error %d" %(err))
        return 0
    else: pass
    # Get data size from file size
    fsize = os.path.getsize(infile)
    dsize = fsize - hsize

    # Add data size to header dict
    hd['data_bytes'] = dsize
    
    return hd

def check_and_sort_files(infiles, reverse=False):
    """
    Check basic consistency of input files 
    
    sort in in ascending order if reverse=False, 
    otherwise in descending order
    """
    # Get header info
    hdlist = []
    for infile in infiles:
        hd_ii = get_cs_info(infile)
        hdlist.append(hd_ii)
    
    freqs = np.array([ hd['fch1'] for hd in hdlist ])
    bws = np.array([ hd['foff'] for hd in hdlist ])
    tstarts = np.array([ hd['tstart'] for hd in hdlist ])
    srcs = np.array([ hd['source_name'] for hd in hdlist ])
    dsizes = np.array([ hd['data_bytes'] for hd in hdlist ])
    nbits = np.array([ hd['nbits'] for hd in hdlist ])

    # check consistency
    err = 0 
    if len( np.unique(bws) ) > 1:
        print("Multiple BWs:")
        print(np.unique(bws))
        err += 1
    else: pass

    if len( np.unique(srcs) ) > 1:
        print("Multiple source names")
        print(np.unique(srcs))
        err += 1
    else: pass

    if len( np.unique(tstarts) ) > 1:
        print("Multiple MJD start dates")
        print(np.unique(tstarts))
        err += 1
    else: pass

    if len( np.unique(dsizes) ) > 1:
        print("Multiple data sizes")
        print(dsizes)
        err += 1
    
    if len( np.unique(nbits) ) > 1:
        print("Multiple data bit values")
        print(nbits)
        err += 1

    if err:
        print("Files incompatible")
        return 0
    else: pass

    # Now sort files by frequency
    xx = np.argsort(freqs)
    
    # Flip if required
    if reverse:
        xx = xx[::-1]
    else: pass

    freqsort = np.array([ freqs[xxi] for xxi in xx ])
    filesort = [infiles[xxi] for xxi in xx]

    # Get global pars
    src = srcs[0]
    bw  = bws[0]
    tstart = tstarts[0]
    dsize = dsizes[0]
    nbit = nbits[0]

    return filesort, freqsort, src, bw, tstart, dsize, nbit


def check_pol_data(check_out1, check_out2):
    """
    check the `check output` for both pols to 
    make sure metadata is consistent
    """
    f1, freqs1, src1, bw1, t1, d1, b1 = check_out1
    f2, freqs2, src2, bw2, t2, d2, b2 = check_out2

    retval = 1

    # check freqs
    if not np.all(freqs1==freqs2):
        print("Pols have diff freqs!")
        print(freqs1, freqs2)
        retval *= 0
    
    # check bw
    if bw1 != bw2:
        print("Pols have different bandwidths!")
        print(bw1, bw2)
        retval *= 0

    # check times
    if t1 != t2:
        print("Pols have different start times!")
        print(t1, t2)
        retval *= 0

    # check data sizes 
    if d1 != d2:
        print("Pols have different data sizes!")
        print(d1, d2)
        retval *= 0

    # check byte size
    if b1 != b2:
        print("Pols have different byte sizes!")
        retval *=0 

    return retval

     

def read_cs(infile, start=0, count=-1):
    """
    Read complex voltage data

    start = sample number to start at

    count = number of samples to read after start
    """
    # Get header info
    hd, hsize, err = fb.read_header(infile, 4, fb.fmtdict)
    hdr_size = hsize
   
    # Get bit size and set dtype
    nbits = hd['nbits']
    if nbits == 8:
        dtype = 'int8'
    elif nbits == 32:
        dtype = 'float32'
    else:
        print("Only supports nbits of 8 or 32")
        print("nbits = %d invalid" %(nbits))
        return
   
    # Offset in bytes
    offset = hdr_size + (start * 2 * (nbits//8))

    # Samples to read in 
    ncount = -1
    if count >= 0:
        ncount = 2 * count
    else: pass

    # Open file in read mode
    with open(infile, 'rb') as fin:
        dat = np.fromfile(fin, offset=offset, count=ncount, dtype=dtype)

    return dat


def read_many_cs(infiles, start=0, count=-1):
    """
    Read complex voltage data from many files
    """
    # If count is -1, read one file to get count
    if count < 0:
        dtmp = read_cs(infiles[0], start=start, count=-1)
        count = len(dtmp) // 2
    else: pass

    # Read header from first file to get data type
    hd, hsize, err = fb.read_header(infiles[0], 4, fb.fmtdict)
    
    nbits = hd['nbits']
    if nbits == 8:
        dtype = 'int8'
    elif nbits == 32:
        dtype = 'float32'
    else:
        print("Only supports nbits of 8 or 32")
        print("nbits = %d invalid" %(nbits))
        return
    
    N = len(infiles)
    dd_all = np.zeros( (N, 2 * count), dtype=dtype)

    for ii in range(N):
        fii = infiles[ii]
        dii = read_cs(fii, start=start, count=count)
        dd_all[ii, :] = dii[:]

    return dd_all


##############################
##  WRITE TO DADA FILE(S)   ##
##############################

def write_dada_header(outfile, fcenter_MHz, bw_MHz, tsamp_us, 
                      mjd_start, nchan, Nt_all, bps=8, npol=1, 
                      source_name=None):
    """
    Write the dada header 

    Nt_all is total number of time samples for final 
    DADA file
    """
    # Make header object
    hdr = DADA_HDR()

    # File size is number of bytes in payload (not header)
    # (N time samps) * (n pols ) * (N chans) * (2 cv of size bps/8 bytes)
    hdr.file_size = Nt_all * npol * nchan * 2 * (bps / 8)
    
    # Source name if available
    if source_name is not None:
        hdr.source_name = source_name

    # Bits per sample
    hdr.nbit = bps

    # Number of channels
    hdr.nchan = nchan
    
    # Number of pols
    hdr.npol = npol

    # Sample time in microseconds
    hdr.tsamp = tsamp_us

    # Set the total bandwidth in MHz
    hdr.bw = bw_MHz

    # UTC Start Time
    tstart = Time(mjd_start, format='mjd')
    tstr = str(tstart.datetime64)
    utc_str = '-'.join(tstr.split('T'))
    hdr.utc_start = utc_str
    
    # MJD start
    hdr.mjd_start = mjd_start
    
    # Center Frequency in MHz
    hdr.freq = fcenter_MHz

    # Now that the header is set, we can write it to file
    hdr.write_header(outfile)

    return hdr


def write_to_dada(outfile, data, fcenter_MHz, bw_MHz, tsamp_us, 
                  mjd_start, nchan, source_name=None, fac=10):
    """
    Data in shape (2 * Nt, Nchan)

    Note: we are fixing bits per sample = 8 and npol = 1
    """
    # Check data shape
    Nt_all = data.shape[0] // 2
    if data.shape[1] == nchan:
        pass
    else:
        print("Data shape mismatch")
        print("%d chans in data instead of %d" %(data.shape[1], nchan))
        return 0

    # Scale data 
    if data.dtype == 'float32':
        data = data * fac
    else: pass

    # Get data in the right format 
    data = np.reshape(data, (Nt_all, 2, nchan))
    data = np.swapaxes(data, 1, 2)
    data = np.reshape(data, (Nt_all, 2 * nchan))

    if data.dtype == 'int8':
        dd_out = data
    else:
        dd_out = data.astype('int8')

    # Write header
    hdr = write_dada_header(outfile, fcenter_MHz, bw_MHz, tsamp_us, 
                            mjd_start, nchan, Nt_all, 
                            source_name=source_name)

    # Now we can append the data to this file
    with open(outfile, 'ab') as fout:
        fout.write(dd_out)
    
    # and that should do it
    return 


def repack_pols(dat1, dat2):
    """
    Take data from each polarization in the 
    format (2 * Nt, Nchan) and combine and reshape
    into one array of shape (2 * Nt, 2 pol, Nchan)
    that can be written to DADA
    """
    # Make sure they have the same shape
    if dat1.shape != dat2.shape:
        print("Shape mismatch in pols:")
        print(dat1.shape)
        print(dat2.shape)
        return
    else: pass

    Nt = dat1.shape[0] // 2 
    Nc = dat1.shape[1]
    Np = 2

    # input array shapes = (2Nt, Nc)
    dd = np.concatenate((dat1, dat2), axis=0)

    # dd shape = (Np * 2Nt, Nc)
    dd = np.reshape(dd, (Np, Nt, 2, Nc))
    
    # dd shape = (Np, Nt, 2, Nc)
    dd = np.swapaxes(dd, 2, 3)

    # dd shape = (Np, Nt, Nc, 2)
    dd = np.swapaxes(dd, 0, 1)
    
    # dd shape = (Nt, Np, Nc, 2)
    dd = np.reshape(dd, (Nt, Np, 2 * Nc))
    
    # Final shape = (Nt, Np, 2*Nc)
    return dd
    

def append_to_dada(outfile, data, fac=10):
    """
    Data in shape (2 * Nt, Nchan)

    Note: we are fixing output bits per sample = 8 and npol = 1
    """
    # Apply a scaling factor to better rep as 8-bit if float
    if data.dtype == 'float32':
        data = data * fac
    else: pass

    nchan = data.shape[1]
    
    # Get data in the right format 
    data = np.reshape(data, (-1, 2, nchan))
    data = np.swapaxes(data, 1, 2)
    data = np.reshape(data, (-1, 2 * nchan))

    if data.dtype == 'int8':
        dd_out = data
    else:
        dd_out = data.astype('int8')

    # Now we can append the data to this file
    with open(outfile, 'ab') as fout:
        fout.write(dd_out)
    
    # and that should do it
    return 


def append_to_dada_pol(outfile, data, fac=10):
    """
    Should already be in appropriate shape

    Note: we are fixing output bits per sample = 8 
    """
    # Apply a scaling factor to better rep as 8-bit if float
    if data.dtype == 'float32':
        data = data * fac
    else: pass

    if data.dtype == 'int8':
        dd_out = data
    else:
        dd_out = data.astype('int8')

    # Now we can append the data to this file
    with open(outfile, 'ab') as fout:
        fout.write(dd_out.flatten())
    
    # and that should do it
    return 

#############################
##  CS -> Dynamic Spectrum  #
#############################

def cs2dada_multipass(basename, indir, outdir, mem_lim_gb=32.0):
    """
    Get CS baseband files of the form:

        {basename}*.cs

    in "indir" directory, and convert 
    them to a DADA file of the form 

        basename.dada

    in the directory "outdir"
    """
    t0 = time.time()
    # Get files 
    infiles = glob.glob("%s/%s*.cs" %(indir, basename))
   
    # Check files and get info 
    check_out = check_and_sort_files(infiles, reverse=True)  

    if check_out == 0:
        return 0
    else:
        pass

    sfiles, freqs, src, bw, tstart, dsize, nbits = check_out
    
    # Get nec info for DADA 
    nchan = len(freqs)
    full_bw_MHz = -1 * np.abs(bw) * nchan
    fcenter_MHz = np.mean(freqs)
    tsamp_us = 1.0 / np.abs(bw)
    mjd_start = tstart
    Nt_total = int( dsize / (2 * nbits / 8) )
    
    # Write data as (nspec, nchan) shape 
    dada_out = "%s/%s.dada" %(outdir, basename)

    # Write header
    hdr = write_dada_header(dada_out, fcenter_MHz, full_bw_MHz, 
                            tsamp_us, mjd_start, nchan, Nt_total, 
                            source_name=src)

    # Figure out the number of samples per file per chunk
    Nsamp_per_chunk = int( 0.5 * 0.75 * 10**9 * mem_lim_gb / (2 * (nbits//8) * nchan) )

    # How many steps to read all the data?
    nsteps = int( np.ceil( Nt_total / Nsamp_per_chunk ) )

    # Read data from files in chunks and append to dada file
    fac = 10
    start = 0 
    for ii in range(nsteps):
        if ii == nsteps-1:
            count = Nt_total - start
        else:
            count = Nsamp_per_chunk

        print("   Step %d / %d" %(ii+1, nsteps))
        
        dd_chunk = read_many_cs(sfiles, start, count=count)
        append_to_dada(dada_out, dd_chunk.T, fac=fac)
        start += count
    
    t1 = time.time()
    print("DADA conversion -- %.1f seconds" %(t1-t0))
    return


def cs2dada_multipass_pol(lcp_base, rcp_base, indir, outdir, 
                          outbase, mem_lim_gb=32.0):
    """
    Get CS baseband files of the form:

        {l/rcp_base}*.cs

    in "indir" directory, and convert 
    them to a DADA file of the form 

        basename.dada

    in the directory "outdir"
    """
    t0 = time.time()
    # Get files 
    lcp_files = glob.glob("%s/%s*.cs" %(indir, lcp_base))
    rcp_files = glob.glob("%s/%s*.cs" %(indir, rcp_base))
   
    # Check files and get info 
    lcp_check_out = check_and_sort_files(lcp_files, reverse=True)  
    rcp_check_out = check_and_sort_files(rcp_files, reverse=True)  

    # Make sure there's something there
    if (lcp_check_out == 0) or (rcp_check_out == 0):
        print('empty')
        return 0
    else: pass

    # Make sure metadata is consistent between pols
    if not check_pol_data(lcp_check_out, rcp_check_out):
        return 0
    
    lcp_sfiles, freqs, src, bw, tstart, dsize, nbits = lcp_check_out
    rcp_sfiles, _, _, _, _, _, _ = rcp_check_out
    
    # Get nec info for DADA 
    nchan = len(freqs)
    full_bw_MHz = -1 * np.abs(bw) * nchan
    fcenter_MHz = np.mean(freqs)
    tsamp_us = 1.0 / np.abs(bw)
    mjd_start = tstart
    Nt_total = int( dsize / (2 * nbits / 8) )
    npol = 2
    
    # Write data as (nspec, nchan) shape 
    dada_out = "%s/%s.dada" %(outdir, outbase)

    # Write header
    hdr = write_dada_header(dada_out, fcenter_MHz, full_bw_MHz, 
                            tsamp_us, mjd_start, nchan, Nt_total, 
                            npol=npol, source_name=src)

    # Figure out the number of samples per file per chunk
    Nsamp_per_chunk = int( 0.5 * 0.75 * 10**9 * mem_lim_gb / (npol * 2 * (nbits//8) * nchan) )

    # How many steps to read all the data?
    nsteps = int( np.ceil( Nt_total / Nsamp_per_chunk ) )

    # Read data from files in chunks and append to dada file
    fac = 10
    start = 0 
    for ii in range(nsteps):
        if ii == nsteps-1:
            count = Nt_total - start
        else:
            count = Nsamp_per_chunk

        print("   Step %d / %d" %(ii+1, nsteps))
       
        print("       reading lcp...")
        dd_lcp = read_many_cs(lcp_sfiles, start, count=count)
        print("       reading rcp...")
        dd_rcp = read_many_cs(rcp_sfiles, start, count=count)

        print("       reshaping...")
        dd = repack_pols(dd_lcp.T, dd_rcp.T)
    
        print("       writing...")
        append_to_dada_pol(dada_out, dd, fac=fac)
        start += count
    
    t1 = time.time()
    print("DADA conversion -- %.1f seconds" %(t1-t0))
    return


def run_digifil(infile, outfile, dm, nchan, nthread=1, inc_ddm=True):
    """
    Use digifil to coherently de-disperse and channelize 
    the 7-channel baseband datat in the DADA file

    infile:  DADA file to process

    outfile: name of output *.fil file

    dm:  DM for coherent de-dispersion

    nchan: total number of channels in output filterbank 

    nthreads: Number of threads for proc (default=1)

    inc_ddm: Also do the incoherent de-dispersion? (default=True)
    """
    # Use options to build up command 
    cmd = "digifil"

    # add "-b-32" to output 32-bit floats 
    cmd += " -b-32"

    # add -IO to turn of normalization
    cmd += " -I0"

    # add number of threads
    cmd += " -threads %d" %(nthread)

    # add channelization 
    cmd += " -F %d:D" %(nchan)

    # add dm 
    cmd += " -D %.4f" %(dm)

    # add incoherent flag if required
    if inc_ddm:
        cmd += " -K"
    else: pass

    # add output file name
    cmd += " -o %s" %(outfile)

    # add input file name
    cmd += " %s" %(infile) 

    # Print command
    print(cmd)

    # Run command
    t0 = time.time()
    subprocess.run(cmd, shell=True)
    t1 = time.time()

    # print time
    #print("digifil step -- %.1f seconds" %(t1-t0))
    
    return


def get_chunk_base(basename, cs_dir):
    """
    get the unique {basename}-XXXX values
    """
    # Get the paths
    pfiles = glob.glob("%s/%s*cs" %(cs_dir, basename))
    
    # Get file names 
    fnames = np.array([ pf.split("/")[-1] for pf in pfiles ])

    # Get chunk bases
    cnames = np.array([ fn.rsplit('-', 1)[0] for fn in fnames ])

    # Get sorted list of unique values
    unames = np.unique(cnames)

    return unames   


def cs2fil_multipass(basename, cs_dir, dada_dir, fil_dir, dm, nchan, 
                     mem_lim_gb=16.0, nthread=1, inc_ddm=False):
    """
    Take cs files of form {basename}*.cs in directory cs_dir 
    and convert them into a single DADA file {basename}.dada 
    in dada_dir.

    Then run digifil to produce a coherently de-dispersed 
    filterbank at DM "dm" with a total of nchan channels.
    Use nthread threads for digifil and remove incohrent 
    dispersive delay if inc_ddm=True 
    """
    t0 = time.time()
    # Read in and convert data to DADA file
    cs2dada_multipass(basename, cs_dir, dada_dir, 
                      mem_lim_gb=mem_lim_gb)
    
    t1 = time.time()

    # Run digifil 
    dada_file = "%s/%s.dada" %(dada_dir, basename)
    fil_file = "%s/%s.fil" %(fil_dir, basename)
    run_digifil(dada_file, fil_file, dm, nchan, nthread=nthread, inc_ddm=inc_ddm)
    t2 = time.time()

    # Clean up by removing dada file
    if os.path.exists(fil_file) and os.path.exists(dada_file):
        os.remove(dada_file)

    print("")
    print("Convert to DADA -- %.1f sec" %(t1 - t0))
    print("digifil         -- %.1f sec" %(t2 - t1))
    print("")
    print(" Total = %.1f sec" %(t2-t0))

    return


def check_lcp_rcp_files(lcp_base, rcp_base, cs_dir):
    """
    Get file bases, make sure right number 
    """
    lcp_chunk_bases = get_chunk_base(lcp_base, cs_dir)
    rcp_chunk_bases = get_chunk_base(rcp_base, cs_dir)
    
    Nlcp = len(lcp_chunk_bases)
    Nrcp = len(rcp_chunk_bases)

    retval = 1

    if Nlcp == 0:
        print("No LCP files found with basename %s in %s" \
                                   %(lcp_base, cs_dir))
        retval *= 0

    if Nrcp == 0:
        print("No RCP files found with basename %s in %s" \
                                   %(rcp_base, cs_dir))
        retval *= 0

    if Nlcp != Nrcp:
        print("Found %d LCP files and %d RCP files" %(Nclp, Nrcp))
        retval *= 0

    return retval, lcp_chunk_bases, rcp_chunk_bases
        



@click.command()
@click.option("--lcp_base", type=str, 
              help="Name of LCP cs files: {lcp_base}*.cs")
@click.option("--rcp_base", type=str, 
              help="Name of RCP cs files: {rcp_base}*.cs")
@click.option("--out_base", type=str, 
              help="Basename of output dada file")
@click.option("--cs_dir", type=str,
              help="Directory containing cs files")
@click.option("--dada_dir", type=str, 
              help="Directory for DADA files")
#@click.option("--fil_dir", type=str,
#              help="Directory for *.fil files")
#@click.option("--dm", type=float, 
#              help="DM for intra-channel coherent dedispersion")
#@click.option("--nchan", type=int, 
#              help="Total number of output channels in filterbank")
@click.option("--mem_lim", type=float, default=16.0, 
              help="Max memory to use during DADA conversion (GB)")
#@click.option("--nthread", type=int, default=1,  
#              help="Number of threads for digifil processing")
def main(lcp_base, rcp_base, cs_dir, dada_dir, out_base, mem_lim):
    """
    Convert multiple chunks of complex sampled voltage data 
    in two polarizations to a single DADA file
    
    Input cs files assumed to be in the form 
 
           {basename}-XXXX-YY.cs in directory 
    
    where {basename} is the base name associated with all 
    files, XXXX is a number 0000-9999 that gives the data 
    chunk order, and YY is 01-07 and represents the channel.

    The channels of each time chunk will be combined into a 
    DADA file called 

           {basename}-XXXX.dada
    """
    # First get a list of unique {basename}-XXXX values
    lcp_chunk_bases = get_chunk_base(lcp_base, cs_dir)
    rcp_chunk_bases = get_chunk_base(rcp_base, cs_dir)

    # Make sure we actually have some files
    rval, lcp_chunk_bases, rcp_chunk_bases =\
                 check_lcp_rcp_files(lcp_base, rcp_base, cs_dir)
    if not rval:
        sys.exit()

    for ii in range(len(lcp_chunk_bases)):
        lcp_ii = lcp_chunk_bases[ii]
        rcp_ii = rcp_chunk_bases[ii]
        print("Processing: %s + %s ..." %(lcp_ii, rcp_ii))
        cs2dada_multipass_pol(lcp_ii, rcp_ii, cs_dir, dada_dir, 
                              out_base, mem_lim_gb=mem_lim)
    
    return


debug = 0
    
if __name__ == "__main__":
    if not debug:
        main()

