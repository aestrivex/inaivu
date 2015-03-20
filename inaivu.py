from __future__ import division
import os
import numpy as np
from traits.api import HasTraits, Any, Dict, Instance, Str
from traitsui.api import (View, Item, Group, OKCancelButtons, ShellEditor,
    HGroup, VGroup, Handler)
from traitsui.message import error as error_dialog

import mne
from mayavi import mlab
from mayavi.core.ui.api import MayaviScene, SceneEditor, MlabSceneModel
import surfer
import nibabel as nib
from collections import OrderedDict

import source_signal

class InaivuModel(HasTraits):

    brain = Any # Instance(surfer.viz.Brain)
    ieeg_loc = Any #OrderedDict
    ieeg_glyph = Any # Instance(mlab.Glyph3D)

    scene = Any # mayavi.core.Scene
    scene = Instance(MlabSceneModel, ())

    shell = Dict

    subjects_dir = Str
    subject = Str('fake_subject')

    invasive_signals = Dict # Str -> Instance(InvasiveSignal)
    current_invasive_signal = Instance(source_signal.InvasiveSignal)

    traits_view = View(
        Item('scene', editor=SceneEditor(scene_class=MayaviScene),
            show_label=False, height=500, width=500),
        Item('shell', editor=ShellEditor(), height=300, show_label=False),
        
        title='Das ist meine Wassermelone es ist MEINE',
        resizable=True,
    )

    def build_surface(self, subjects_dir=None, subject=None):
        '''
        creates a pysurfer surface and plots it

        specify subject, subjects_dir or these are taken from the environment

        Returns
        -------
        brain | surfer.viz.Brain
            Pysurfer brain object
        figure | mlab.scene
            Mayavi scene object
        '''

        if subjects_dir==None:
            subjects_dir = os.environ['SUBJECTS_DIR']
        if subject==None:
            subject = os.environ['SUBJECT']

        self.brain = surfer.Brain( subject, hemi='both', surf='pial', 
            figure=self.scene.mayavi_scene, subjects_dir=subjects_dir, 
            curv=False )

        #self.scene = self.brain._figures[0][0]

        self.brain.toggle_toolbars(True)

        #set the surface unpickable
        for srf in self.brain.brains:
            srf._geo_surf.actor.actor.pickable = False
            srf._geo_surf.actor.property.opacity = 0.35

        return self.brain

    def plot_ieeg(self, raw):
        '''
        given a raw .fif file with sEEG electrodes, (and potentially other
        electrodes), extract and plot all of the sEEG electrodes in the file

        Returns
        -------
        ieeg_glyph | mlab.glyph
            Mayavi 3D glyph object
        '''

        ra = mne.io.Raw(raw)

        elecs = [(name, ra.info['chs'][i]['loc'][:3])
            for i,name in enumerate(ra.ch_names) if
            ra.info['chs'][i]['kind'] == mne.io.constants.FIFF.FIFFV_SEEG_CH]

        self.ch_names = [e[0] for e in elecs]
        self.ieeg_loc = dict(elecs)

        locs = np.array([e[1] for e in elecs])

        source = mlab.pipeline.scalar_scatter(locs[:,0], locs[:,1], locs[:,2],
            figure=self.scene.mayavi_scene)
        
        self.ieeg_glyph = mlab.pipeline.glyph( source, scale_mode='none',
            scale_factor=6, mode='sphere', figure=self.scene.mayavi_scene, 
            color=(1,0,0), name='garbonzo', colormap='BuGn')

        #self.ieeg_glyph = mlab.points3d( locs[:,0], locs[:,1], locs[:,2],
        #    color = (1,0,0), scale_factor=6, figure=figure)

        return self.ieeg_glyph

    def plot_ieeg_montage(self, montage):
        '''
        given a raw montage file with sEEG electrode coordinates,
        extract and plot all of the sEEG electrodes in the montage

        Returns
        -------
        ieeg_glyph | mlab.glyph
            Mayavi 3D glyph object
        '''

        mopath = os.path.abspath(montage)
        mo = mne.channels.read_montage(mopath)

        elecs = [(name, pos) for name,pos in zip(mo.ch_names, mo.pos)]
        self.ch_names = mo.ch_names
        self.ieeg_loc = dict(elecs)

        locs = mo.pos

        source = mlab.pipeline.scalar_scatter(locs[:,0],locs[:,1], locs[:,2],
            figure=self.scene.mayavi_scene)

        self.ieeg_glyph = mlab.pipeline.glyph( source, scale_mode='none',
            scale_factor=6, mode='sphere', figure=self.scene.mayavi_scene,
            color=(1,0,0), name='gableebo', colormap='BuGn')

        return self.ieeg_glyph

    def generate_subcortical_surfaces(self, subjects_dir=None, subject=None):
        import subprocess

        if subjects_dir is not None:
            os.environ['SUBJECTS_DIR']=subjects_dir
        if subject is None:
            subject = os.environ['SUBJECT']

        aseg2srf_cmd = os.path.realpath('aseg2srf')

        subprocess.call([aseg2srf_cmd, '-s', subject])

    def viz_subcortical_surfaces(self, subjects_dir=None, subject=None):
        structures_list = {
                           'hippocampus': ([53, 17], (.69, .65, .93)),
                           'amgydala': ([54, 18], (.8, .5, .29)),
                           'thalamus': ([49, 10], (.318, 1, .447)),
                           'caudate': ([50, 11], (1, .855, .67)),
                           'putamen': ([51, 12], (0, .55, 1)),
                           'insula': ([55, 19], (1, 1, 1)),
                           'accumbens': ([58, 26], (1, .44, 1)),
                                                     }

        for (strucl, strucr), _ in structures_list.values():
            
            for strucu in (strucl, strucr):
                surf_file = os.path.join(subjects_dir, subject,
                    'ascii', 'aseg_%03d.srf'%strucu )

                if not os.path.exists(surf_file):
                    continue

                v, tri = mne.read_surface(surf_file)

                mlab.triangular_mesh( v[:,0], v[:,1], v[:,2], tri,
                    opacity = .35, 
                    color=(.5, .5, .5))
                    #)

    def viz_subcortical_points(self, subjects_dir=None, subject=None):
        '''
        add transparent voxel structures at the subcortical structures
        '''

        if subjects_dir is None:
            subjects_dir = os.environ['SUBJECTS_DIR']
        if subject is None:
            subject = os.environ['SUBJECT']

        structures_list = {
                           'hippocampus': ([53, 17], (.69, .65, .93)),
                           'amgydala': ([54, 18], (.8, .5, .29)),
                           'thalamus': ([49, 10], (.318, 1, .447)),
                           'caudate': ([50, 11], (1, .855, .67)),
                           'putamen': ([51, 12], (0, .55, 1)),
                           'insula': ([55, 19], (1, 1, 1)),
                           'accumbens': ([58, 26], (1, .44, 1)),
                                                     }

        asegf = os.path.join( subjects_dir, subject, 'mri', 'aseg.mgz')
        aseg = nib.load( asegf )
        asegd = aseg.get_data()

        for struct in structures_list:
            (strucl, strucr), color = structures_list[struct]
            
            for strucu in (strucl, strucr):

                strucw = np.where(asegd==strucu)

                if np.size(strucw) == 0:
                    print 'Nonne skippy %s' % struct
                    continue

                import geometry as geo
                xfm = geo.get_vox2rasxfm(asegf, stem='vox2ras-tkr')
                strucd = np.array(geo.apply_affine( np.transpose(strucw), xfm ))

                print np.shape(strucd)

                src = mlab.pipeline.scalar_scatter( strucd[:,0], strucd[:,1], 
                    strucd[:,2], figure=self.scene.mayavi_scene )

                mlab.pipeline.glyph( src, scale_mode='none', 
                    scale_factor=0.4, mode='sphere', opacity=1,
                    figure=self.scene.mayavi_scene, color=color )
            
    
    def add_invasive_signal(self, name, signal):
        #self.invasive_signals.append(signal)
        self.invasive_signals[name] = signal
        self.current_invasive_signal = signal

    def set_current_invasive_signal(self, name):
        self.current_invasive_signal = self.invasive_signals[name]

    def _display_invasive_signal_timepoint(self, idx, ifunc):
        '''
        Currently assumes that the sampling frequency is always 1
        '''
        #if idx%1 == 0:
        if False:
            #scalars = [self.invasive_signal.signals_dict[ch][idx] for i,ch in 
            #    enumerate(self.ch_names)]

            scalars = self.current_invasive_signal.mne_source_estimate.data[:,
                idx]
        else:
            scalars = ifunc(idx)

        #print idx
        #import pdb
        #pdb.set_trace()

        self.ieeg_glyph.mlab_source.dataset.point_data.scalars = (
            np.array(scalars))
        self.ieeg_glyph.actor.mapper.scalar_visibility = True

        mlab.draw(figure=self.scene.mayavi_scene)

    def movie(self, movname, invasive=True, noninvasive=True,
              framerate=24, interpolation='quadratic', dilation=2,
              tmin=None, tmax=None, normalization='none'):
        from scipy.interpolate import interp1d

        if not invasive and not noninvasive:
            raise ValueError("That movie is not interesting")

        if noninvasive:
            raise NotImplementedError("noninvasive signals not supported yet")

        if invasive:
            if self.current_invasive_signal is None:
                raise ValueError("No signal provided")
            if self.current_invasive_signal.mne_source_estimate is None:
                raise ValueError("Signal has no source estimate")


        stc = self.current_invasive_signal.mne_source_estimate

        sample_rate = stc.tstep

        if tmin is None:
            tmin = stc.tmin

        if tmax is None:
            tmax = self.current_invasive_signal.mne_source_estimate.times[-1]

        smin = np.argmin(np.abs(stc.times - tmin))
        smax = np.argmin(np.abs(stc.times - tmax))

        #print tmin, tmax, stc.times
        #print smin, smax

        # catch if the user asked for invasive timepoints that dont exist
        if tmin < stc.tmin:
            raise ValueError("Time window too low for invasive signal")
        if tmax > stc.times[-1]:
            raise ValueError("Time window too high for invasive signal")


        time_length = tmax-tmin
        sample_length = smax-smin+1

        tstep_size = 1 / (framerate * dilation)
        sstep_size = tstep_size / sample_rate
    
        if time_length % sstep_size == 0:
            sstop = smax + sstep_size / 2
        else:
            sstop = smax

        movie_sample_times = np.arange(smin, sstop, sstep_size)

        raw_sample_times = np.arange(smin, smax+1)

        #print raw_sample_times.shape
        #print movie_sample_times.shape

        #steps = sample_length*step_factor + 1

        ##times = np.linspace(0, length, length, endpoint=False)
        ##all_times = np.linspace(0, length, steps, endpoint=False)
        #all_times = np.linspace(0, sample_length-1, steps, endpoint=True)
        exact_times = np.arange(sample_length)
        #exact_times = np.arange( len(self.current_invasive_signal.
        #    mne_source_estimate.times) )

        #all_times = interp1d(movie_sample_times, exact_times)

        #this interpolation is exactly linear
        all_times = interp1d(raw_sample_times, 
            exact_times)(movie_sample_times)

        steps = len(all_times)

        #print exact_times.shape
        #print all_times.shape

        #print steps, len(exact_times)
         
        from tempfile import mkdtemp
        tempdir = mkdtemp()
        frame_pattern = 'frame%%0%id.png' % (np.floor(np.log10(steps)) + 1)
        fname_pattern = os.path.join(tempdir, frame_pattern)

        images_written = []

        #data = np.array(
        #    [self.current_invasive_signal.signals_dict[ch][tmin:tmax]
        #    for ch in self.ch_names])
        data = self.current_invasive_signal.mne_source_estimate.data[:,
            smin:smax+1]
        #print exact_times.shape, data.shape

        if normalization=='none':
            pass
        elif normalization=='conservative':
            dmax = np.max(self.current_invasive_signal.mne_source_estimate.
                data)
            dmin = np.min(self.current_invasive_signal.mne_source_estimate.
                data)
            data = (data-dmin) / (dmax-dmin)
        elif normalization=='local': 
            dmax = np.max(data)
            dmin = np.min(data)
            data = (data-dmin) / (dmax-dmin)

        ifunc = interp1d( exact_times , data, interpolation, axis=1)

        for i,idx in enumerate(all_times):
            frname = fname_pattern % i

            #do the data display method
            self._display_invasive_signal_timepoint(idx, ifunc)
            #VIEW + INTERPOLATION NOT CORRECT

            self.brain.save_image(frname)

        #return images_written
        from surfer.utils import ffmpeg
        ffmpeg(movname, fname_pattern, framerate=framerate, codec=None)
        
if __name__=='__main__':
    im = InaivuModel()
    im.configure_traits()
