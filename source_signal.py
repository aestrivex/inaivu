from __future__ import division

from traits.api import HasTraits, Float, Dict, Int, Any, List
import numpy as np
from scipy import io
import mne
import nibabel as nib
import os
from error_dialog import error_dialog

def adj_sort(cur_ord, desired_ord):
    if len(cur_ord) < len(desired_ord):
        error_dialog("Wrong number of electrodes")
    keys={}
    for i,k in enumerate(cur_ord):
        keys[k]=i
    reorder_map = map(keys.get, desired_ord)
    if None in reorder_map:
        error_dialog("Failed to map all values, check input")
    return reorder_map 

class SourceSignal(HasTraits):
    mne_source_estimate = Any #Instance(mne._BaseSourceEstimate)
    data = Any #Instance(np.ndarray), data formed to correct order

class NoninvasiveSignal(SourceSignal):
    pass

class InvasiveSignal(SourceSignal):
    mne_source_estimate = Any #Instance(mne._BaseSourceEstimate)
    ix_pos_map = Any #Instance(np.ndarray)
    ch_names = List

def load_stc(stcf):
    return mne.read_source_estimate(stcf)

def signal_from_stc(stc, ordering=None, invasive=False):
    if invasive:
        sig = InvasiveSignal()
        if ordering is not None:
            sig.ix_pos_map, sig.ch_names = load_ordering_file(ordering)
    else:
        sig = NoninvasiveSignal()

    sig.mne_source_estimate = stc
        
    return sig

def load_montage(montage_file):
    import os
    return mne.channels.read_montage(os.path.realpath(montage_file))

def load_ordering_file(ordering_file):
    #if already a list, just return it back
    if type(ordering_file) == list:
        return np.arange(len(ordering_file)), ordering_file

    ixes = []
    names = []

    with open(ordering_file) as fd:
        for i,ln in enumerate(fd):
            ln = ln.strip()
            if ln=='delete':
                continue
            ixes.append(i)
            names.append(ln)

    return ixes, names
read_ordering_file = load_ordering_file
    
def save_ordering_file(fname, ordering):
    with open(ordering, 'w') as fd:
        for name in ordering:
            fd.write('%s\n'%name)

def gen_stupid_gamma_signal(ch_names, hemi='rh'):
    ivs = InvasiveSignal()
    ivs.ch_names = ch_names

    if hemi=='rh':
        vertnos = [np.array(()), np.arange(len(ch_names))]
    else:
        vertnos = [np.arange(len(ch_names)), np.array(())]

    from scipy.stats import gamma

    tsignal = np.zeros((len(ch_names), 100))
    for i,ch_name in enumerate(ch_names):
        scale_param = np.random.randint(5,15)
        multi_param = np.random.randint(1,4)
        min_param = np.random.randint(10)
        max_param = np.random.randint(10,15)

        #import pdb
        #pdb.set_trace()

        tsignal[i,:] = np.array([gamma.pdf(j,scale_param)*multi_param for
            j in np.linspace(min_param,max_param,100)])

    stc = mne.SourceEstimate(tsignal, tmin=1, tstep=1, vertices=vertnos,
        subject='woethiezluok')

    ivs.mne_source_estimate = stc

    return ivs

def gen_stupid_sinusoidal_signal(ch_names, hemi='rh'):
    ivs = InvasiveSignal()
    ivs.ch_names = ch_names
    
    if hemi=='rh':
        vertnos = [np.array(()), np.arange(len(ch_names))]
    else:
        vertnos = [np.arange(len(ch_names)), np.array()]

    tsignal = np.zeros((len(ch_names), 100)) 
    for i, ch_name in enumerate(ch_names):

        amp = np.random.random()*.5 
        freqp = np.random.random()*(5-.2)+.2
        phasep = np.random.random()*2*np.pi
        funcp = [np.sin, np.cos][int(np.random.randint(2))]

        tsignal[i,:] = np.array([amp*funcp(2*np.pi*freqp*j+phasep)+.5 for j
            in np.arange(100)])

    stc = mne.SourceEstimate(tsignal, tmin=1, tstep=1, vertices=vertnos,
        subject='woethiezluok')

    ivs.mne_source_estimate = stc

    return ivs

def stc_from_fiff(fiff_file, names=None):
    '''
    Creates a source estimate from the sensor space channels in a fiff file
    '''
    ra = mne.io.Raw(fiff_file)

    tmin = ra.index_as_time(0)
    tstep = 1/ra.info['sfreq']

    if names is not None:
        vertnos = np.squeeze([np.where(np.array(ra.ch_names)==name) for name in
            np.intersect1d(ra.ch_names, names)])

    else:
        #use all names
        vertnos = np.arange(len(ra.ch_names))

    ordering_names = np.array(ra.ch_names)
    ordering_names[ np.setdiff1d( np.arange(len(ra.ch_names)), vertnos)] = 'del'

    #convert to 'delete' without messing with datatypes
    ordering_names = np.array(map(lambda x:'delete' if x=='del' else x,
        ordering_names.tolist()))

    #by arbitrary choice we use the RH no matter where are the electrodes
    stc = mne.SourceEstimate(ra[:][0][np.sort(vertnos)],
        vertices = [np.array(()), np.sort(vertnos)], tmin=tmin, tstep=tstep)

    return stc, ordering_names

def textfile_from_fsfast_nifti(fsfast_nifti_file, outfile):
    res = nib.load(fsfast_nifti_file)
    resd = res.get_data()
    np.savetxt(outfile, np.squeeze(res).T)

def stc_from_fsfast_nifti(fsfast_nifti_file, tr, hemi=None):
    res = nib.load(fsfast_nifti_file)
    resd = np.squeeze(res.get_data())
    return _stc_from_array(resd, tr, fsfast_nifti_file, hemi=hemi)

def stc_from_two_fsfast_niftis(lh_nifti, rh_nifti, tr):
    res_lh = nib.load(lh_nifti)
    res_rh = nib.load(rh_nifti)
    resd_lh = np.squeeze(res_lh.get_data())
    resd_rh = np.squeeze(res_rh.get_data())
    return _stc_from_bihemi_array(resd_lh, resd_rh, tr)

def stc_from_text_file(text_file, tr, hemi=None):
    data = np.loadtxt(text_file)
    return _stc_from_array(data, tr, text_file, hemi=hemi)

def build_bihemi_stc(lh_stc, rh_stc):
    lh_data = lh_stc.data
    rh_data = rh_stc.data

    lh_verts = lh_stc.vertices[0]
    rh_verts = rh_stc.vertices[1]

    lh_tmin = lh_stc.tmin
    rh_tmin = rh_stc.tmin

    lh_tstep = lh_stc.tstep
    rh_tstep = rh_stc.tstep

    if lh_tmin != rh_tmin or lh_tstep != rh_tstep:
        error_dialog("Timing must be consistent in left and right stc")

    stc = mne.SourceEstimate( np.vstack((lh_data, rh_data)),
        vertices=[lh_verts, rh_verts], tmin=lh_tmin, tstep=lh_tstep)

    return stc

def _stc_from_array(data, tr, filename, hemi=None, tmin=0):
    #TODO allow bihemi stc
    nvert, ntimes = data.shape

    if hemi==None:
        #try to read hemi from filename
        lh = (filename.find('lh') != -1)
        rh = (filename.find('rh') != -1)

        if lh and not rh:
            hemi = 'lh'
        if rh and not lh:
            hemi = 'rh'

    if hemi not in ('lh','rh'):
        error_dialog('Correct hemisphere not provided and could not figure it'
            ' out')

    if hemi=='lh':
        vertices = [np.arange(nvert), np.array(())]
    else:
        vertices = [np.array(()), np.arange(nvert)]

    stc = mne.SourceEstimate( data, vertices=vertices, tmin=tmin, tstep=tr )

    return stc

def _stc_from_bihemi_array(lh_data, rh_data, tr, filename, tmin=0):
    lh_nvert, lh_ntimes = lh_data.shape
    rh_nvert, rh_ntimes = rh_data.shape

    if lh_ntimes != rh_ntimes:
        error_dialog("Inconsistent timing across hemispheres")

    data = np.vstack((lh_data, rh_data))
    vertices = [np.arange(lh_nvert), np.arange(rh_nvert)]

    stc = mne.SourceEstimate( data, vertices=vertices, tmin=tmin, tstep=tr )

    return stc

def create_signal_from_fieldtrip_stclike(ft_file, source_field, 
    time_field='Time', ordering=None, name_field=None, invasive=False,
    hemi=None, tmin=0, tr=.001):
    '''
    Extract a signal that is essentially an STC signal
    '''
    ftd = io.loadmat(ft_file)

    if name_field is None and ordering is None:
        error_dialog("Need to specify either a field or file or list with "
            "channel order") 


    if invasive:
        sig = InvasiveSignal()

        if ordering is not None:
            sig.ix_pos_map, sig.ch_names = load_ordering_file(ordering)

        elif name_field is not None:
            sig.ch_names = ftd[name_field]
            sig.ix_pos_map = np.arange(len(sig.ch_names))

        if len(ftd[source_field]) != len(sig.ch_names):
            print len(ftd[source_field]), len(sig.ch_names)
            error_dialog("Incorrect number of electrodes")

    else:
        sig = NoninvasiveSignal()

        #maybe some stuff


    tr = np.diff(np.squeeze(ftd[time_field]))[0]

    from PyQt4.QtCore import pyqtRemoveInputHook
    import pdb
    pyqtRemoveInputHook()
    #pdb.set_trace()
    stc = _stc_from_array( ftd[source_field], tr, ft_file, hemi=hemi, 
        tmin=tmin )
    sig.mne_source_estimate = stc

    return sig 


def read_labels_from_cell_array_in_struct(mfile, field, labels_info):
    ftd = io.loadmat(mfile)
    labels = {}
    for col_num, label_name in labels_info.iteritems():
        labels[label_name] = map(str, map(lambda x:x[col_num][0], ftd[field]))
    return labels

# this function does not operate on well formulated fieldtrip files
# consequently it does no processing that we should trust
def create_signal_from_fieldtrip_timefqchan(ft_file):
    ftd = io.loadmat(ft_file)

    nr_frequencies = ftd['datad']['trial'][0,0].size

    channels = map(str, map(lambda x:x[0][0], ftd['ElectrodeLabels']))
    nr_channels = ftd['datad']['trial'][0,0][0,0].size

    for i in xrange(nr_frequencies):

        ivs = InvasiveSignal()

        ivs.sampling_rate = float(ftd['datad']['fsample']) 

        ivs.time_start = float(ftd['datad'][0,0][0][0][0][0])
        ivs.time_end = float(ftd['datad'][0,0][0][0][-1][0])

        tsignal = {}
        
        for j, ch in enumerate(channels):
            tsignal[ch] = ftd['ChanFreqTimeMatrix'][j,i,:]
        
        ivs.signals_dict = tsignal

        yield ivs

def identify_roi_from_atlas(pos, approx=4, atlas=None, subjects_dir=None,
    subject=None ):
    '''
    Find the surface labels contacted by an electrode at this position
    in RAS space.

    Parameters
    ----------

    pos : np.ndarray
        1x3 matrix holding position of the electrode to identify
    approx : int
        Number of millimeters error radius
    atlas : str or None
        The string containing the name of the surface parcellation,
        does not apply to subcortical structures. If None, aparc is used.
    '''
    if subjects_dir is None or subjects_dir=='':
        subjects_dir = os.environ['SUBJECTS_DIR']
    if subject is None or subject=='':
        subject = os.environ['SUBJECT']

    if atlas is None or atlas in ('', 'aparc'):
        return identify_roi_from_aparc(pos, approx=approx,
            subjects_dir=subjects_dir, subject=subject)

    from scipy.spatial.distance import cdist
    # conceptually, we should grow the closest vertex around this electrode
    # probably following snapping but the code for this function is not
    # altered either way

    # load the surfaces and annotation
    # uses the pial surface, this change is pushed to MNE python

    lh_pia, _ = nib.freesurfer.read_geometry(
        os.path.join(subjects_dir, subject, 'surf', 'lh.pial'))

    rh_pia, _ = nib.freesurfer.read_geometry(
        os.path.join(subjects_dir, subject, 'surf', 'rh.pial'))

    pia = np.vstack((lh_pia, rh_pia))

    # find closest vertex
    # import pdb
    # pdb.set_trace()
    closest_vert = np.argmin(cdist(pia, [pos]))

    # grow the area of surface surrounding the vertex
    import mne

    # we force the label to only contact one hemisphere even if it is
    # beyond the extent of the medial surface
    hemi_str = 'lh' if closest_vert<len(lh_pia) else 'rh'
    hemi_code = 0 if hemi_str=='lh' else 1

    if hemi_str == 'rh':
        closest_vert -= len(lh_pia)

    radius_label, = mne.grow_labels(subject, closest_vert, approx, hemi_code,
        subjects_dir=subjects_dir, surface='pial')

    parcels = mne.read_labels_from_annot(subject, parc=atlas, hemi=hemi_str,
        subjects_dir=subjects_dir, surf_name='pial')

    regions = []
    for parcel in parcels:
        if len(np.intersect1d(parcel.vertices, radius_label.vertices))>0:
            #force convert from unicode
            # regions.append(str(parcel.name))
            regions.append(parcel)

    # subcortical_regions = identify_roi_from_aparc(pos, approx=approx,
    #     subjects_dir=subjects_dir, subject=subject, subcortical_only=True)
    #
    # if regions is not None and subcortical_regions is not None:
    #     regions.extend(subcortical_regions)

    return regions

def identify_roi_from_aparc( pos, approx=4, subjects_dir=None, subject=None,
    subcortical_only = False):
    '''
    Find the volumetric labels contacted by an electrode at this position
    in RAS space.

    Parameters
    ----------

    pos : np.ndarray
        1x3 matrix holding position of the electrode to identify
    approx : int
        Number of millimeters error radius
    subcortical_only : bool
        if True, exclude cortical labels
    '''
    if subjects_dir is None or subjects_dir=='':
        subjects_dir = os.environ['SUBJECTS_DIR']
    if subject is None or subject=='':
        subject = os.environ['SUBJECT']

    def find_neighboring_regions(pos, mri_dat, region, approx, excludes):
        spot_sz = int(np.around(approx * 2 + 1))
        x, y, z = np.meshgrid(range(spot_sz), range(spot_sz), range(spot_sz))

        # approx is in units of millimeters as long as we use the RAS space
        # segmentation
        neighb = np.vstack((np.reshape(x, (1, spot_sz ** 3)),
            np.reshape(y, (1, spot_sz ** 3)),
            np.reshape(z, (1, spot_sz ** 3)))).T - approx

        regions = []

        #import pdb
        #pdb.set_trace()

        for p in xrange(neighb.shape[0]):
            cx, cy, cz = (pos[0]+neighb[p,0], pos[1]+neighb[p,1],
                pos[2]+neighb[p,2])
            d_type = mri_dat[cx, cy, cz]
            label_index = region['index'].index(d_type)
            regions.append(region['label'][label_index])

        if excludes:
            from re import compile
            excluded = compile('|'.join(excludes))
            regions = [x for x in regions if not excluded.search(x)]

        return np.unique(regions).tolist()
