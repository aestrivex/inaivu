from __future__ import division
import os
import numpy as np
from traits.api import HasTraits, Any, Dict, Instance, Str, Float
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

    noninvasive_signals = Dict # Str -> Instance(NoninvasiveSignal)
    current_noninvasive_signal = Instance(source_signal.NoninvasiveSignal)

    opacity = Float(.35)

    smoothl_mat = Any #Either(np.ndarray, None)
    smoothr_mat = Any #Either(np.ndarray, None)

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
            srf._geo_surf.actor.property.opacity = self.opacity

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

    def add_noninvasive_signal(self, name, signal):
        self.noninvasive_signals[name] = signal
        self.current_noninvasive_signal = signal

    def set_current_noninvasive_signal(self, name):
        self.current_noninvasive_signal = self.noninvasive_signals[name]

    def _display_invasive_signal_timepoint(self, idx, ifunc):
        '''
        Currently assumes that the sampling frequency is always 1
        '''
        #if idx%1 == 0:
        #if False:
        #    scalars = self.current_invasive_signal.mne_source_estimate.data[:,
        #        idx]
        #else:
        scalars = ifunc(idx)

        self.ieeg_glyph.mlab_source.dataset.point_data.scalars = (
            np.array(scalars))
        self.ieeg_glyph.actor.mapper.scalar_visibility = True

    def _display_noninvasive_signal_timepoint(self, idx, ifunc, 
            interpolation='quadratic'):
        #    scalars = (self.current_noninvasive_signal.mne_source_estimate.
        #        data[:,idx])
        #else:
        scalars = ifunc(idx)
        #print idx

        lvt = self.current_noninvasive_signal.mne_source_estimate.lh_vertno
        rvt = self.current_noninvasive_signal.mne_source_estimate.rh_vertno

        print np.shape(scalars)

        print lvt
        print rvt
        
        if len(lvt) > 0:
            lh_scalar = scalars[lvt]
            lh_surf = self.brain.brains[0]._geo_surf
            if len(lvt) < len(self.brain.geo['lh'].coords):
                lh_scalar = self.smoothl * lh_scalar

            lh_surf.mlab_source.scalars = lh_scalar

        if len(rvt) > 0:
            rh_scalar = scalars[rvt]
            rh_surf = self.brain.brains[1]._geo_surf
            if len(rvt) < len(self.brain.geo['rh'].coords):
                rh_scalar = self.smoothr * rh_scalar

            rh_surf.mlab_source.scalars = rh_scalar

        #self.brain.set_data_time_index(idx, interpolation)

    def movie(self, movname, invasive=True, noninvasive=True,
              framerate=24, interpolation='quadratic', dilation=2,
              tmin=None, tmax=None, normalization='none', debug_labels=False,
              smoothing_steps=20):
        #potentially worth providing different options for normalization and
        #interpolation for noninvasive and invasive data


        if not invasive and not noninvasive:
            raise ValueError("That movie is not interesting")

        if noninvasive:
            if self.current_noninvasive_signal is None:
                raise ValueError("No signal provided")
            if self.current_noninvasive_signal.mne_source_estimate is None:
                raise ValueError("Signal has no source estimate")

            ni_times, _, nfunc = self._create_movie_samples(
                self.current_noninvasive_signal, tmin=tmin, tmax=tmax,
                framerate=framerate, dilation=dilation,
                interpolation=interpolation, normalization=normalization,
                is_invasive=False)

            nsteps = len(ni_times)
            steps = nsteps

            lvt = self.current_noninvasive_signal.mne_source_estimate.lh_vertno
            rvt = self.current_noninvasive_signal.mne_source_estimate.rh_vertno

            if 0 < len(lvt) < len(self.brain.geo['lh'].coords):
                ladj = surfer.utils.mesh_edges(self.brain.geo['lh'].faces)
                self.smoothl = surfer.utils.smoothing_matrix(lvt, ladj, 
                    smoothing_steps)

            if 0 < len(rvt) < len(sefl.brain.geo['rh'].coords):
                radj = surfer.utils.mesh_edges(self.brain.geo['lh'].faces)
                self.smoothr = surfer.utils.smoothing_matrix(lvt, ladj,
                    smoothing_steps)

            for brain in self.brain.brains:
                brain._geo_surf.module_manager.scalar_lut_manager.lut_mode = (
                    'RdBu')
                brain._geo_surf.module_manager.scalar_lut_manager.reverse_lut=(
                    True)
                brain._geo_surf.actor.mapper.scalar_visibility=True

            if len(lvt) > 0:
                lh_surf = self.brain.brains[0]._geo_surf
            if len(rvt) > 0:
                rh_surf = self.brain.brains[1]._geo_surf

#            time_label = 'diebhog thyme %d' if debug_labels else None
#            if (np.size(self.current_noninvasive_signal.mne_source_estimate.
#                lh_vertno) > 1):
#                    self.brain.add_data( self.current_noninvasive_signal.
#                        mne_source_estimate.lh_data, hemi='lh',
#                        alpha=self.opacity, time_label=time_label )
#            if (np.size(self.current_noninvasive_signal.mne_source_estimate.
#                rh_vertno) > 1):
#                    self.brain.add_data( self.current_noninvasive_signal.
#                        mne_source_estimate.rh_data, hemi='rh', 
#                        alpha=self.opacity, time_label=time_label ) 

        if invasive:
            if self.current_invasive_signal is None:
                raise ValueError("No signal provided")
            if self.current_invasive_signal.mne_source_estimate is None:
                raise ValueError("Signal has no source estimate")

            i_times, _, ifunc = self._create_movie_samples(
                self.current_invasive_signal, tmin=tmin, tmax=tmax,
                framerate=framerate, dilation=dilation,
                interpolation=interpolation, normalization=normalization,
                is_invasive=True)

            isteps = len(i_times)
            steps = isteps

        if noninvasive and invasive:
            if isteps != nsteps:
                raise ValueError("Bad sampling")
             

        from tempfile import mkdtemp
        tempdir = mkdtemp()
        frame_pattern = 'frame%%0%id.png' % (
            np.floor(np.log10(steps)) + 1)
        fname_pattern = os.path.join(tempdir, frame_pattern)

        images_written = []

        for i in xrange(steps):
            frname = fname_pattern % i

            #do the data display method
            if invasive:
                iidx = i_times[i]
                self._display_invasive_signal_timepoint(iidx, ifunc)
        
            if noninvasive:
                nidx = ni_times[i]
                self._display_noninvasive_signal_timepoint(nidx, nfunc,
                    interpolation=interpolation)

            mlab.draw(figure=self.scene.mayavi_scene)
            self.brain.save_image(frname)

        #return images_written
        from surfer.utils import ffmpeg
        ffmpeg(movname, fname_pattern, framerate=framerate, codec=None)

    def _create_movie_samples(self, sig, framerate=24, 
            interpolation='quadratic',
            dilation=2, tmin=None, tmax=None, normalization='none',
            is_invasive=False):

        from scipy.interpolate import interp1d

        stc = sig.mne_source_estimate

        sample_rate = stc.tstep

        if tmin is None:
            tmin = stc.tmin

        if tmax is None:
            tmax = (stc.times[-1])

        smin = np.argmin(np.abs(stc.times - tmin))
        smax = np.argmin(np.abs(stc.times - tmax))

        # catch if the user asked for invasive timepoints that dont exist
        if tmin < stc.tmin:
            raise ValueError("Time window too low for %s signal" %
                'invasive' if is_invasive else 'noninvasive')
        if tmax > stc.times[-1]:
            raise ValueError("Time window too high for %s signal" %
                'invasive' if is_invasive else 'noninvasive')

        time_length = tmax-tmin
        sample_length = smax-smin+1

        tstep_size = 1 / (framerate * dilation)
        sstep_size = tstep_size / sample_rate
    
        if np.allclose(sample_length % sstep_size, 0, atol=sample_rate/2):
            sstop = smax + sstep_size / 2
        else:
            sstop = smax

        if np.allclose(time_length % tstep_size, 0):
            tstop = tmax + tstep_size / 2
        else:
            tstop = tmax

        movie_sample_times = np.arange(smin, smax, sstep_size)
        raw_sample_times = np.arange(smin, smax+1)

        exact_times = np.arange(sample_length)

        #this interpolation is exactly linear
        all_times = interp1d(raw_sample_times, 
            exact_times)(movie_sample_times)

        data = stc.data[:,smin:smax+1]

        print data.shape
        print all_times.shape
        print sstep_size, tstep_size
        print tmin, tmax, tstop, smin, smax, sstop
        print sample_rate

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

        interp_func = interp1d( exact_times , data, interpolation, axis=1)

        return all_times, data, interp_func
        
if __name__=='__main__':
    #force Qt to relay ctrl+C
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    im = InaivuModel()
    im.configure_traits()
