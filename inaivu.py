from traits.api import HasTraits, Any, Dict, Instance

import os
import mne
import numpy as np
from mayavi import mlab
import surfer
from collections import OrderedDict

import timesignal

class InaivuModel(HasTraits):

    brain = Any # Instance(surfer.viz.Brain)
    ieeg_loc = Any #OrderedDict
    ieeg_glyph = Any # Instance(mlab.Glyph3D)

    scene = Any # mayavi.core.Scene

    invasive_signal = Instance(timesignal.InvasiveSignal)

    def build_surface(self, figure=None, subjects_dir=None, subject=None):
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
            figure=figure, subjects_dir=subjects_dir, curv=False )

        self.scene = self.brain._figures[0][0]

        self.brain.toggle_toolbars(True)

        #set the surface unpickable
        for srf in self.brain.brains:
            srf._geo_surf.actor.actor.pickable = False
            srf._geo_surf.actor.property.opacity = 0.35

        return self.brain, self.scene

    def plot_ieeg(self, raw, figure=None):
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

        source = mlab.pipeline.scalar_scatter( locs[:,0], locs[:,1], locs[:,2],
            figure=figure)
        
        self.ieeg_glyph = mlab.pipeline.glyph( source, scale_mode='none',
            scale_factor=6, mode='sphere', figure=figure, color=(1,0,0),
            name='garbonzo', colormap='BuGn')

        #self.ieeg_glyph = mlab.points3d( locs[:,0], locs[:,1], locs[:,2],
        #    color = (1,0,0), scale_factor=6, figure=figure)

        return self.ieeg_glyph
    
    def show(self):
        mlab.show()

    def add_invasive_signal(self, sfreq, signaldict):
        self.invasive_signal = timesignal.InvasiveSignal(sampling_rate = sfreq,
            signals_dict=signaldict, 
            length=len(signaldict[signaldict.keys()[0]]))

    def display_invasive_signal_timepoint(self, idx, ifunc):
        '''
        Currently assumes that the sampling frequency is always 1
        '''
        #if idx%1 == 0:
        if False:
            scalars = [self.invasive_signal.signals_dict[ch][idx] for i,ch in 
                enumerate(self.ch_names)]
        else:
            scalars = ifunc(idx)

        print idx
        #import pdb
        #pdb.set_trace()

        self.ieeg_glyph.mlab_source.dataset.point_data.scalars = (
            np.array(scalars))
        self.ieeg_glyph.actor.mapper.scalar_visibility = True

        #mlab.draw(figure=self.scene)

    def movie(self, movname, invasive=True, noninvasive=True, step_factor=4.,
              framerate=24, interpolation='quadratic',
              tmin=None, tmax=None):
        from scipy.interpolate import interp1d

        if not invasive and not noninvasive:
            raise ValueError("That movie is not interesting")

        if noninvasive:
            raise NotImplementedError("noninvasive signals not supported yet")

        if invasive:
            if self.invasive_signal is None:
                raise ValueError("No signal provided")

        # catch things like, noninvasive and invasive signals being misaligned
        # in their temporal extent yet the user asked for them both


        # for now just plot an invasive signal as stupidly as possible

        length = self.invasive_signal.length
        steps = (length-1)*step_factor + 1

        #times = np.linspace(0, length, length, endpoint=False)
        #all_times = np.linspace(0, length, steps, endpoint=False)
        all_times = np.linspace(0, length-1, steps, endpoint=True)
        exact_times = np.arange(length)
         
        from tempfile import mkdtemp
        tempdir = mkdtemp()
        frame_pattern = 'frame%%0%id.png' % (np.floor(np.log10(steps)) + 1)
        fname_pattern = os.path.join(tempdir, frame_pattern)

        images_written = []

        data = np.array([self.invasive_signal.signals_dict[ch] for ch in
            self.ch_names])
        print exact_times.shape, data.shape
        ifunc = interp1d( exact_times, data, interpolation, axis=1)

        for idx in all_times:
            frname = fname_pattern % idx

            #do the data display method
            self.display_invasive_signal_timepoint(idx, ifunc)
            #VIEW + INTERPOLATION NOT CORRECT

            self.brain.save_image(frname)

        #return images_written
        from surfer.utils import ffmpeg
        ffmpeg(movname, fname_pattern, framerate=framerate, codec=None)
        
