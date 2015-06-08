from __future__ import division
import os
import numpy as np
from traits.api import (HasTraits, Any, Dict, Instance, Str, Float,
    Range, on_trait_change, File, Button, Int, Bool, Enum)
from traitsui.api import (View, Item, Group, OKCancelButtons, ShellEditor,
    HGroup, VGroup, Handler, RangeEditor, Action, CancelButton, Handler,
    NullEditor, Label)
from error_dialog import error_dialog

import mne
from mayavi import mlab
from mayavi.core.ui.api import MayaviScene, SceneEditor, MlabSceneModel
import surfer
import nibabel as nib
from collections import OrderedDict

import source_signal

class InaivuModel(Handler):

    brain = Any # Instance(surfer.viz.Brain)
    ieeg_loc = Any #OrderedDict
    ieeg_glyph = Any # Instance(mlab.Glyph3D)
    ch_names = Any #List(Str) ? 

    scene = Any # mayavi.core.Scene
    scene = Instance(MlabSceneModel, ())

    _time_low = Float(0)
    _time_high = Float(1)
    time_slider = Float(0)

    shell = Dict

    subjects_dir = Str
    subject = Str('fake_subject')

    invasive_signals = Dict # Str -> Instance(InvasiveSignal)
    current_invasive_signal = Instance(source_signal.InvasiveSignal)

    noninvasive_signals = Dict # Str -> Instance(NoninvasiveSignal)
    current_noninvasive_signal = Instance(source_signal.NoninvasiveSignal)

    opacity = Float(.35)

    use_smoothing = Bool(False)
    smoothing_steps = Int(0)

    smoothl = Any #Either(np.ndarray, None)
    smoothr = Any #Either(np.ndarray, None)

    browser = Any #Instance(BrowseStc)

    current_script_file = File
    run_script_button = Button('Run script')


    # movie window
    make_movie_button = Button('Movie')

    movie_filename = File
    movie_normalization_style = Enum('local', 'global', 'none')

    movie_use_invasive = Bool(True)
    movie_use_noninvasive = Bool(True)
    movie_tmin = Float(0.)
    movie_tmax = Float(1.)
    movie_invasive_tmin = Float(0.)
    movie_invasive_tmax = Float(1.)
    movie_noninvasive_tmin = Float(0.)
    movie_noninvasive_tmax = Float(1.)

    movie_framerate = Float(24)
    movie_dilation = Float(2)
    movie_bitrate = Str('750k')
    movie_interpolation = Enum('quadratic', 'cubic', 'linear', 'slinear',
        'nearest', 'zero')
    movie_animation_degrees = Float(0.)
    movie_sample_which_first = Enum('invasive', 'noninvasive')

    OKMakeMovieAction = Action(name='Make movie', action='do_movie')

    traits_view = View(
        Item('scene', editor=SceneEditor(scene_class=MayaviScene),
            show_label=False, height=300, width=300),
        VGroup(
            Item('time_slider', 
                editor=RangeEditor(mode='xslider', low_name='_time_low',
                    high_name='_time_high', format='%.3f', is_float=True), 
                label='time'),
        ),
        HGroup(
            Item('make_movie_button', show_label=False),
            Item('current_script_file'),
            Item('run_script_button', show_label=False),
        ),
        #Item('time_slider', style='custom', show_label=False),
        Item('shell', editor=ShellEditor(), height=300, show_label=False),
        
        title='Das ist meine Wassermelone es ist MEINE',
        resizable=True,
    )

    def _run_script_button_fired(self):
        with open(self.current_script_file) as fd:
            exec(fd)

    make_movie_view = View(
        Label('Click make movie to specify filename'),
        #Item('movie_filename', label='filename', style='readonly'),
        HGroup(
            VGroup(
                HGroup(
                    Item('movie_use_invasive', 
                        label='include invasive signal'),
                ),
                Item('movie_invasive_tmin', label='invasive tmin',
                    enabled_when="movie_use_invasive"),
                Item('movie_invasive_tmax', label='invasive tmin',
                    enabled_when="movie_use_invasive"),
                Item('movie_normalization_style', label='normalization'),
                Item('movie_sample_which_first', label='samples first',
                    enabled_when="movie_use_noninvasive and " 
                        "movie_use_invasive"),
            ),
            VGroup(
                HGroup(
                    Item('movie_use_noninvasive', 
                        label='include noninvasive signal'),
                ),
                Item('movie_noninvasive_tmin', label='noninvasive tmin',
                    enabled_when="movie_use_noninvasive"),
                Item('movie_noninvasive_tmax', label='noninvasive_tmax',
                    enabled_when="movie_use_noninvasive"),
                Item('movie_interpolation', label='interp'),
            ),
            VGroup(
                Item('movie_bitrate', label='bitrate'),
                Item('movie_framerate', label='framerate'),
                Item('movie_dilation', label='temporal dilation'),
                Item('movie_animation_degrees', label='degrees rotation'),
            ),
        ),

#        HGroup(
#            Item('movie_tmin', label='tmin'),
#            Item('movie_tmax', label='tmax'),
#            Item('movie_dilation', label='temporal dilation'),
#        ),
#        HGroup(
#            Item('movie_framerate', label='framerate'),
#            Item('movie_bitrate', label='bitrate (b/s)'),
#            Item('movie_interpolation', label='interp'),
#        ),
        title='Chimer exodus from Aldmeris',
        buttons=[OKMakeMovieAction, CancelButton],
    )

    def _make_movie_button_fired(self):
        self.edit_traits(view='make_movie_view')

    def do_movie(self, info):
        from pyface.api import FileDialog, OK as FileOK
        dialog = FileDialog(action='save as')
        dialog.open()
        if dialog.return_code != FileOK:
            return

        self.movie_filename = os.path.join(dialog.directory, dialog.filename)
        info.ui.dispose()
        self.movie( self.movie_filename, 
            noninvasive_tmin=self.movie_noninvasive_tmin,
            noninvasive_tmax=self.movie_noninvasive_tmax,
            invasive_tmin=self.movie_invasive_tmin,
            invasive_tmax=self.movie_invasive_tmax,
            normalization=self.movie_normalization_style,
            framerate=self.movie_framerate,
            dilation=self.movie_dilation,
            bitrate=self.movie_bitrate,
            interpolation=self.movie_interpolation,
            animation_degrees=self.movie_animation_degrees,
            samples_first=self.movie_sample_which_first,
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

    def invasive_callback(self, picker):
        if picker.actor not in self.ieeg_glyph.actor.actors:
            return

        ptid = int(picker.point_id / self.ieeg_glyph.glyph.glyph_source.
            glyph_source.output.points.to_array().shape[0])

        from browse_stc import do_browse
        # todo: change this to the real roi surface signal
        surface_signal_rois = np.random.randn(*self.current_invasive_signal.mne_source_estimate.data.shape)
        if self.browser is None or self.browser.figure is None:
            self.browser = do_browse(self.current_invasive_signal, 
                bads=['LPT8'], n_channels=1, const_event_time=2.0,
                surface_signal_rois=surface_signal_rois,
                glyph = self.ieeg_glyph)
#        elif self.browser.figure is None:
#            self.browser = do_browse(self.current_invasive_signal, 
#                bads=['LPT8'], n_channels=1,
#                                     const_event_time=2.0)

        pt_loc = tuple(self.ieeg_glyph.mlab_source.points[ptid])
        pt_name = self.ieeg_loc[pt_loc]
        pt_index = self.ch_names.index(pt_name)
        #print ptid, pt_loc, pt_name, pt_index

        self.browser._plot_imitate_scroll(pt_index)
        
    def plot_ieeg(self, raw=None, montage=None, elec_locs=None, 
            ch_names=None):
        '''
        given a raw .fif file with sEEG electrodes, (and potentially other
        electrodes), extract and plot all of the sEEG electrodes in the file.
        
        alternately accepts a list of electrode locations and channel names
        or a montage file

        Returns
        -------
        ieeg_glyph | mlab.glyph
            Mayavi 3D glyph object
        '''
        if raw is None and montage is None and (
                elec_locs is None or ch_names is None):
            error_dialog("must specify raw .fif file or list of electrode "
                "coordinates and channel names")

        if raw is not None:
            ra = mne.io.Raw(raw)

            #elecs = [(name, ra.info['chs'][i]['loc'][:3])
            elecs = [(tuple(ra.info['chs'][i]['loc'][:3]), name)
                for i,name in enumerate(ra.ch_names) if
                ra.info['chs'][i]['kind'] == mne.io.constants.FIFF.FIFFV_SEEG_CH]

            self.ch_names = [e[1] for e in elecs]

            locs = np.array([e[0] for e in elecs])

        elif montage is not None:
            sfp = source_signal.load_montage(montage)
            locs = np.array(sfp.pos)
            self.ch_names = sfp.ch_names

            elecs = [(tuple(loc), name) for name, loc in zip(self.ch_names, locs)]

        else:
            locs = np.array(elec_locs)
            
            self.ch_names = ch_names

            elecs = [(tuple(loc), name) for name, loc in zip(ch_names, locs)]

        # compare signal.ch_names to the ch_names here

        self.ieeg_loc = dict(elecs)

        source = mlab.pipeline.scalar_scatter(locs[:,0], locs[:,1], locs[:,2],
            figure=self.scene.mayavi_scene)
        
        self.ieeg_glyph = mlab.pipeline.glyph( source, scale_mode='none',
            scale_factor=6, mode='sphere', figure=self.scene.mayavi_scene, 
            color=(1,0,0), name='garbonzo', colormap='BuGn')

        #self.ieeg_glyph = mlab.points3d( locs[:,0], locs[:,1], locs[:,2],
        #    color = (1,0,0), scale_factor=6, figure=figure)

        self.ieeg_glyph.mlab_source.dataset.point_data.scalars = np.zeros(
            (len(locs),))
        self._force_render()

        pick = self.scene.mayavi_scene.on_mouse_pick(self.invasive_callback)
        pick.tolerance = .1

        return self.ieeg_glyph

    def interactivize_ieeg(self):
        from browse_stc import do_browse
        self.browser = do_browse( self.current_invasive_signal.
            mne_source_estimate)

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

                surf = mlab.triangular_mesh( v[:,0], v[:,1], v[:,2], tri,
                    opacity = .35, 
                    color=(.5, .5, .5))
                    #)

                surf.actor.actor.pickable = False

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
        if len(self.ch_names) == 0:
            error_dialog("Cannot add invasive signal without first "
                "specifying order of invasive electrodes")

        self.invasive_signals[name] = signal
        self.set_current_invasive_signal(name)

    def set_current_invasive_signal(self, name):
        self.current_invasive_signal = sig = self.invasive_signals[name]

        stc = sig.mne_source_estimate

        if stc.times[0] < self._time_low:
            self._time_low = stc.times[0]
        if stc.times[-1] > self._time_high:
            self._time_high = stc.times[-1]

        #reorder signal in terms of self.ch_names
        reorder_map = source_signal.adj_sort( sig.ch_names, self.ch_names )
        reorder_signal = stc.data[reorder_map]
        #ignore vertices, we only ever bother to get stc.data
        #reorder_vertices = stc.vertno[???]

        sig.ch_names = self.ch_names
        sig.data = reorder_signal

    def add_noninvasive_signal(self, name, signal):
        self.noninvasive_signals[name] = signal
        self.set_current_noninvasive_signal(name)

    def set_current_noninvasive_signal(self, name):
        self.current_noninvasive_signal = sig = self.noninvasive_signals[name]

        stc = sig.mne_source_estimate

        if stc.times[0] < self._time_low:
            self._time_low = stc.times[0]
        if stc.times[-1] > self._time_high:
            self._time_high = stc.times[-1]

        sig.data = stc.data

    def _display_interpolated_invasive_signal_timepoint(self, idx, ifunc):
        scalars = ifunc(idx)

        self.ieeg_glyph.mlab_source.dataset.point_data.scalars = (
            np.array(scalars))
        self.ieeg_glyph.actor.mapper.scalar_visibility = True
        self.ieeg_glyph.module_manager.scalar_lut_manager.data_range = (0,1)

    def _display_interpolated_noninvasive_signal_timepoint(self, idx, ifunc, 
            interpolation='quadratic'):
        scalars = ifunc(idx)

        lvt = self.current_noninvasive_signal.mne_source_estimate.lh_vertno
        rvt = self.current_noninvasive_signal.mne_source_estimate.rh_vertno

        if len(lvt) > 0:
            #assumes all lh scalar precede all rh scalar
            #if not, then we need separate lh_ifunc and rh_ifunc for this case
            lh_scalar = scalars[len(lvt):]
            #lh_scalar = scalars[lvt]
            lh_surf = self.brain.brains[0]._geo_surf
            if len(lvt) < len(self.brain.geo['lh'].coords): 
                if self.smoothing_steps > 0:
                    lh_scalar = self.smoothl * lh_scalar
                else:
                    ls = np.ones( len(self.brain.geo['lh'].coords))*.5
                    ls[rvt] = lh_scalar
                    lh_scalar = ls

            lh_surf.mlab_source.scalars = lh_scalar
            lh_surf.module_manager.scalar_lut_manager.data_range = (0,1)

        if len(rvt) > 0:
            rh_scalar = scalars[:len(rvt)]
            #rh_scalar = scalars[rvt]
            rh_surf = self.brain.brains[1]._geo_surf
            if len(rvt) < len(self.brain.geo['rh'].coords):
                if self.smoothing_steps > 0:
                    rh_scalar = self.smoothr * rh_scalar
                else:
                    rs = np.ones( len(self.brain.geo['rh'].coords))*.5
                    rs[rvt] = rh_scalar
                    rh_scalar = rs

            rh_surf.mlab_source.scalars = rh_scalar
            rh_surf.module_manager.scalar_lut_manager.data_range = (0,1)

        #self.brain.set_data_time_index(idx, interpolation)

    @on_trait_change('time_slider')
    def _show_closest_timepoint_listen(self):
        self.set_closest_timepoint(self.time_slider)
        self._force_render()

    def set_closest_timepoint(self, time, invasive=True, noninvasive=True):
        if noninvasive:
            self._set_noninvasive_timepoint(time)
        if invasive:
            self._set_invasive_timepoint(time)

    def _set_invasive_timepoint(self, t, normalization='global'):
        if self.current_invasive_signal is None:
            return
        sig = self.current_invasive_signal
        stc = sig.mne_source_estimate

        sample_time = np.argmin(np.abs(stc.times - t))

        if normalization=='global':
            dmax = np.max(sig.data)
            dmin = np.min(sig.data)
            data = (sig.data-dmin) / (dmax-dmin)
        else:
            data = sig.data

        scalars = data[:,sample_time]

        #from PyQt4.QtCore import pyqtRemoveInputHook
        #import pdb
        #pyqtRemoveInputHook()
        #pdb.set_trace()

        #unset any changes to the LUT
        self.ieeg_glyph.module_manager.scalar_lut_manager.lut_mode = 'black-white'
        self.ieeg_glyph.module_manager.scalar_lut_manager.lut_mode = 'BuGn'

        self.ieeg_glyph.mlab_source.dataset.point_data.scalars = (
            np.array(scalars))
        self.ieeg_glyph.actor.mapper.scalar_visibility = True
        self.ieeg_glyph.module_manager.scalar_lut_manager.data_range = (0,1)

    def _setup_noninvasive_viz(self):
        lvt = self.current_noninvasive_signal.mne_source_estimate.lh_vertno
        rvt = self.current_noninvasive_signal.mne_source_estimate.rh_vertno

        if (0 < len(lvt) < len(self.brain.geo['lh'].coords) and
                self.smoothing_steps > 0):
            ladj = surfer.utils.mesh_edges(self.brain.geo['lh'].faces)
            self.smoothl = surfer.utils.smoothing_matrix(lvt, ladj, 
                self.smoothing_steps)

        if (0 < len(rvt) < len(self.brain.geo['rh'].coords) and
                self.smoothing_steps > 0):
            radj = surfer.utils.mesh_edges(self.brain.geo['lh'].faces)
            self.smoothr = surfer.utils.smoothing_matrix(rvt, radj,
                self.smoothing_steps)

        for i,brain in enumerate(self.brain.brains):

            #leave gray if no vertices in hemisphere
            if len(lvt)==i==0 or len(rvt)==0==i-1:
                brain._geo_surf.actor.mapper.scalar_visibility=False
                continue

            brain._geo_surf.module_manager.scalar_lut_manager.lut_mode = (
                'RdBu')
            brain._geo_surf.module_manager.scalar_lut_manager.reverse_lut=(
                True)
            brain._geo_surf.actor.mapper.scalar_visibility=True

            brain._geo_surf.module_manager.scalar_lut_manager.data_range=(0,1)

    def _set_noninvasive_timepoint(self, t, normalization='global'):
        if self.current_noninvasive_signal is None:
            return
        stc = self.current_noninvasive_signal.mne_source_estimate

        sample_time = np.argmin(np.abs(stc.times - t))

        lvt = self.current_noninvasive_signal.mne_source_estimate.lh_vertno
        rvt = self.current_noninvasive_signal.mne_source_estimate.rh_vertno

        if normalization=='global':
            if len(lvt) > 0:
                lh_dmax = np.max(stc.lh_data)
                lh_dmin = np.min(stc.lh_data)
                lh_scalar = (stc.lh_data-lh_dmin) / (lh_dmax-lh_dmin)
            if len(rvt) > 0:
                rh_dmax = np.max(stc.rh_data)
                rh_dmin = np.min(stc.rh_data)
                rh_scalar = (stc.rh_data-rh_dmin) / (rh_dmax-rh_dmin)
        else:
            lh_scalar = stc.lh_data
            rh_scalar = stc.rh_data

        self._setup_noninvasive_viz()
    
        if len(lvt) > 0:
            lh_surf = self.brain.brains[0]._geo_surf
            if len(lvt) < len(self.brain.geo['lh'].coords):
                if self.smoothing_steps > 0:
                    lh_scalar = self.smoothl * lh_scalar
                else:
                    ls = np.array( len(self.brain.geo['lh'].coords))
                    ls[lvt] = lh_scalar
                    lh_scalar = ls
            lh_surf.mlab_source.scalars = lh_scalar
            rh_surf.module_manager.scalar_lut_manager.data_range = (0,1)

        if len(rvt) > 0:
            rh_surf = self.brain.brains[1]._geo_surf
            if len(rvt) < len(self.brain.geo['rh'].coords):
                if self.smoothing_steps > 0:
                    rh_scalar = self.smoothr * rh_scalar
                else:
                    rs = np.ones( len(self.brain.geo['rh'].coords))*.5
                    rs[rvt] = rh_scalar
                    rh_scalar = rs
            rh_surf.mlab_source.scalars = rh_scalar
            rh_surf.module_manager.scalar_lut_manager.data_range = (0,1)

    def movie(self, movname, invasive=True, noninvasive=True,
              framerate=24, interpolation='quadratic', dilation=2,
              normalization='local', debug_labels=False,
              bitrate='750k', animation_degrees=0,
              invasive_tmin=None, invasive_tmax=None,
              noninvasive_tmin=None, noninvasive_tmax=None,
              samples_first='invasive'):
        #potentially worth providing different options for normalization and
        #interpolation for noninvasive and invasive data


        if not invasive and not noninvasive:
            error_dialog("Movie has no noninvasive or invasive signals")

        def noninvasive_sampling(nr_samples):
            if self.current_noninvasive_signal is None:
                error_dialog("No noninvasive signal found")
            if self.current_noninvasive_signal.mne_source_estimate is None:
                error_dialog("Noninvasive signal has no source estimate")

            ni_times, _, nfunc, nr_samples = self._create_movie_samples(
                self.current_noninvasive_signal, tmin=noninvasive_tmin,   
                tmax=noninvasive_tmax,
                framerate=framerate, dilation=dilation,
                interpolation=interpolation, normalization=normalization,
                is_invasive=False, nr_samples=nr_samples)

            nsteps = len(ni_times)
            steps = nsteps

            self._setup_noninvasive_viz()

            return ni_times, nfunc, nr_samples, nsteps, steps

        def invasive_sampling(nr_samples):
            if self.current_invasive_signal is None:
                error_dialog("No invasive signal found")
            if self.current_invasive_signal.mne_source_estimate is None:
                error_dialog("Invasive signal has no source estimate")

            i_times, _, ifunc, nr_samples = self._create_movie_samples(
                self.current_invasive_signal, tmin=invasive_tmin, 
                tmax=invasive_tmax,
                framerate=framerate, dilation=dilation,
                interpolation=interpolation, normalization=normalization,
                is_invasive=True, nr_samples=nr_samples)

            isteps = len(i_times)
            steps = isteps

            return i_times, ifunc, nr_samples, isteps, steps

        #ensure that the samples are collected in the right order
        nr_samples = -1

        if invasive and samples_first=='invasive':
            i_times, ifunc, nr_samples, isteps, steps = (
                invasive_sampling(nr_samples))

        if noninvasive:
            ni_times, nfunc, nr_samples, nsteps, steps = (
                noninvasive_sampling(nr_samples))

        if invasive and samples_first!='invasive':
            i_times, ifunc, nr_samples, isteps, steps = (
                invasive_sampling(nr_samples))

        if noninvasive and invasive:
            if isteps != nsteps:
                error_dialog("Movie parameters do not yield equal number of "
                    "samples in invasive and noninvasive timecourses.\n"
                    "Invasive samples: %i\nNoninvasive samples: %i"%(
                    isteps,nsteps))
             

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
                self._display_interpolated_invasive_signal_timepoint(
                    iidx, ifunc)
        
            if noninvasive:
                nidx = ni_times[i]
                self._display_interpolated_noninvasive_signal_timepoint(
                    nidx, nfunc, interpolation=interpolation)

            self.scene.camera.azimuth(animation_degrees)

            self.scene.render()
            mlab.draw(figure=self.scene.mayavi_scene)
            self._force_render()
            self.brain.save_image(frname)

        #return images_written
        from surfer.utils import ffmpeg
        ffmpeg(movname, fname_pattern, framerate=framerate, bitrate=bitrate,
            codec=None)

    def _force_render(self):
        from pyface.api import GUI
        _gui = GUI()
        orig_val = _gui.busy
        _gui.set_busy(busy=True)
        _gui.process_events()
        _gui.set_busy(busy=orig_val)
        _gui.process_events()

    def _create_movie_samples(self, sig, framerate=24, 
            interpolation='quadratic',
            dilation=2, tmin=None, tmax=None, normalization='local',
            is_invasive=False, nr_samples=-1):

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
            error_dialog("Time window too low for %s signal" % (
                'invasive' if is_invasive else 'noninvasive'))
        if tmax > stc.times[-1]:
            error_dialog("Time window too high for %s signal" % (
                'invasive' if is_invasive else 'noninvasive'))


        time_length = tmax-tmin
        sample_length = smax-smin+1

        tstep_size = 1 / (framerate * dilation)
        sstep_size = tstep_size / sample_rate
        sstep_ceil = int(np.ceil(sstep_size))
    
        #to calculate the desired number of samples in the time window, 
        #use the code which checks for step size and optionally adds 1 sample
        #however, the other signal might have a different sampling rate and
        #lack the extra sample even if this signal has it exactly.
        #therefore to compromise, don't do this check or try to interpret
        #at all the missing sample, instead use the wrong number of samples
        #and interpolate to the right thing as close as possible.

        #note that this solution has minor temporal instability inherent;
        #that is to say we are losing information beyond interpolation
        #that might be accurately sampled in one or potentially both
        #signals due to the sampling differences between the signals.
        #this is not a practical problem.
        if nr_samples == -1:
            if np.allclose(time_length % tstep_size, 0, atol=tstep_size/2):
                sstop = smax + sstep_size / 2
            else:
                sstop = smax

            nr_samples = len(np.arange(smin, sstop, sstep_size))
        # end

        movie_sample_times = np.linspace(smin, smax, num=nr_samples)

        #smin is the minimum possible sample, the max is smax
        #to get to exactly smax we need to use smax+1
        raw_sample_times = np.arange(smin, smax+1)

        exact_times = np.arange(sample_length)

        #print sstep_size, tstep_size
        #print tmin, tmax, smin, smax
        #print sample_rate
        #print movie_sample_times.shape, raw_sample_times.shape

        #this interpolation is exactly linear
        all_times = interp1d(raw_sample_times, 
            exact_times)(movie_sample_times)

        data = sig.data[:,smin:smax+1]

        #print data.shape
        #print all_times.shape

        if normalization=='none':
            pass
        elif normalization=='conservative':
            dmax = np.max(sig.data)
            dmin = np.min(sig.data)
            data = (data-dmin) / (dmax-dmin)
        elif normalization=='local': 
            dmax = np.max(data)
            dmin = np.min(data)
            data = (data-dmin) / (dmax-dmin)

        #the interpolation is quadratic and therefore does a very bad job
        #with extremely low frequency varying signals. which can happen when
        #plotting something that looks like raw data.
        interp_func = interp1d( exact_times , data, interpolation, axis=1)

        return all_times, data, interp_func, nr_samples
        
if __name__=='__main__':
    #force Qt to relay ctrl+C
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    im = InaivuModel()
    im.configure_traits()
