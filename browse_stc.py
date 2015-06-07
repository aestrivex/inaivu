# Code adapted from MNE python
# Original Authors: Eric Larson <larson.eric.d@gmail.com>
# Original License: Simplified BSD

from __future__ import division

import copy
from functools import partial

import numpy as np
from scipy.signal import butter, filtfilt

from mne.externals.six import string_types
from mne.io.pick import pick_types
from mne.io.proj import setup_proj
from mne.viz.utils import (figure_nobar, _toggle_options, #_mutable_defaults,
                    _toggle_proj, tight_layout)

from traits.api import HasTraits, Instance, Dict, Int, Float
from traitsui.api import View, Item, Handler
from matplotlib.figure import Figure
from mpleditor import MPLFigureEditor

class BrowseStc(Handler):
    figure = Instance(Figure) 
    params = Dict

    current_channel = Int(-1)
    current_channel_color = Float #this is a scalar not a 4-tuple

    traits_view = View(
        Item('figure', editor=MPLFigureEditor(), show_label=False,
            resizable=True),
        resizable=True,
        # title='Slovenian language differential geometry, 4th ed.'
        title = 'Source Browser'
    )

    def closed(self, info, is_ok):
        self.figure = None 

        #restore previous colors
        import color_utils 
        color_utils.change_single_glyph_color(self.params['glyph'],
            self.current_channel, self.current_channel_color)

#def _plot_update_raw_proj(params, bools):
#    """Helper only needs to be called when proj is changed"""
#    inds = np.where(bools)[0]
#    params['info']['projs'] = [copy.deepcopy(params['projs'][ii])
#                               for ii in inds]
#    params['proj_bools'] = bools
#    params['projector'], _ = setup_proj(params['info'], add_eeg_ref=False,
#                                        verbose=False)
#    _update_raw_data(params)
#    params['plot_fun']()


    def _update_raw_data(self):
        """Helper only needs to be called when time or proj is changed"""
        start = self.params['t_start']
        stop = np.argmin(np.abs( self.params['stc'].times - (start + 
            self.params['duration']) ))
        #stop = self.params['raw'].time_as_index(start + self.params['duration'])[0]
        start = np.argmin(np.abs( self.params['stc'].times - start ))
        #start = self.params['raw'].time_as_index(start)[0]
        #data_picks = pick_types(self.params['raw'].info, meg=True, eeg=True)

        #print start, stop, self.params['t_start'], self.params['duration']

        data = self.params['signal'].data[:, start:stop].copy()
        times = self.params['stc'].times[start:stop]

        # remove DC
        if self.params['remove_dc'] is True:
            data -= np.mean(data, axis=1)[:, np.newaxis]
        if self.params['ba'] is not None:
            data = filtfilt(self.params['ba'][0], self.params['ba'][1],
                                        data, axis=1, padlen=0)
        # scale
        # for di in range(data.shape[0]):
        #     data[di] /= 1e-3 #self.params['scalings'][di]
            # stim channels should be hard limited
            #if self.params['types'][di] == 'stim':
            #    data[di] = np.minimum(data[di], 1.0)
        # clip
        if self.params['clipping'] == 'transparent':
            data[np.logical_or(data > 1, data < -1)] = np.nan
        elif self.params['clipping'] == 'clamp':
            data = np.clip(data, -1, 1, data)
        # Remove bad channels from data
        # First check if it wasn't deleted
        if data.shape[0] > len(self.params['ch_names_no_bads']):
            data = np.delete(data, self.params['bad_indices'], 0)

        #import pdb
        #pdb.set_trace()

        assert(data.shape[0] == len(self.params['ch_names_no_bads'])), \
            'The data dimensions is not equal to the channels number'
        self.params['data'] = data
        self.params['times'] = times


    def _layout_raw(self):
        """Set raw figure layout"""
        s = self.params['fig'].get_size_inches()
        scroll_width = 0.33
        hscroll_dist = 0.33
        vscroll_dist = 0.1
        l_border = 1.2
        r_border = 0.1
        t_border = 0.33
        b_border = 0.5

        # only bother trying to reset layout if it's reasonable to do so
        if s[0] < 2 * scroll_width or s[1] < 2 * scroll_width + hscroll_dist:
            return

        # convert to relative units
        scroll_width_x = scroll_width / s[0]
        scroll_width_y = scroll_width / s[1]
        vscroll_dist /= s[0]
        hscroll_dist /= s[1]
        l_border /= s[0]
        r_border /= s[0]
        t_border /= s[1]
        b_border /= s[1]
        # main axis (traces)
        ax_width = 1.0 - scroll_width_x - l_border - r_border - vscroll_dist
        ax_y = hscroll_dist + scroll_width_y + b_border
        ax_height = 1.0 - ax_y - t_border

        #print ax_width, ax_height
        #print s

        self.params['ax'].set_position([l_border, ax_y, ax_width, ax_height])
        # vscroll (channels)
        pos = [ax_width + l_border + vscroll_dist, ax_y,
               scroll_width_x, ax_height]
        self.params['ax_vscroll'].set_position(pos)
        # hscroll (time)
        pos = [l_border, b_border, ax_width, scroll_width_y]
        self.params['ax_hscroll'].set_position(pos)
        # options button
        pos = [l_border + ax_width + vscroll_dist, b_border,
               scroll_width_x, scroll_width_y]
        self.params['ax_button'].set_position(pos)
        self.params['fig'].canvas.draw()


    def _helper_resize(self, event):
        """Helper for resizing"""
        #size = ','.join([str(s) for s in params['fig'].get_size_inches()])
        #set_config('MNE_BROWSE_RAW_SIZE', size)
        self._layout_raw()


    def _pick_bad_channels(self, event):
        """Helper for selecting / dropping bad channels onpick"""
        bads = self.params['bads']

        # trade-off, avoid selecting more than one channel when drifts are 
        # present
        # however for clean data don't click on peaks but on flat segments
        def f(x, y):
            return y(np.mean(x), x.std() * 2)
        for l in event.inaxes.lines:
            ydata = l.get_ydata()
            if not isinstance(ydata, list) and not np.isnan(ydata).any():
                ymin, ymax = f(ydata, np.subtract), f(ydata, np.add)
                if ymin <= event.ydata <= ymax:
                    this_chan = vars(l)['ch_name']
                    if this_chan in self.params['ch_names']:
                        if this_chan not in bads:
                            bads.append(this_chan)
                            l.set_color(self.params['bad_color'])
                            l.set_zorder(-1)
                        else:
                            bads.pop(bads.index(this_chan))
                            l.set_color(vars(l)['def_color'])
                            l.set_zorder(0)
                        break
        else:
            x = np.array([event.xdata] * 2)
            self.params['ax_vertline'].set_data(x, np.array(self.params['ax'].
                get_ylim()))
            self.params['ax_hscroll_vertline'].set_data(x, np.array([0., 1.]))
            self.params['vertline_t'].set_text('%0.3f' % x[0])
        event.canvas.draw()
        # update deep-copied info to persistently draw bads
        self.params['bads'] = bads


    def _mouse_click(self, event):
        """Vertical select callback"""
        if event.inaxes is None or event.button != 1:
            return
        plot_fun = self.params['plot_fun']
        # vertical scrollbar changed
        if event.inaxes == self.params['ax_vscroll']:
            ch_start = max(int(event.ydata) - self.params['n_channels'] // 2, 0)
            if self.params['ch_start'] != ch_start:
                self.params['ch_start'] = ch_start
                plot_fun()
        # horizontal scrollbar changed
        elif event.inaxes == self.params['ax_hscroll']:
            self._plot_raw_time(event.xdata - self.params['duration'] / 2)

        elif event.inaxes == self.params['ax']:
            self._pick_bad_channels(event)


    def _plot_raw_time(self, value):
        """Deal with changed time value"""
        #info = self.params['info']
        max_times = self.params['n_times'] / self.params['sfreq'] - self.params['duration']
        if value > max_times:
            value = self.params['n_times'] / self.params['sfreq'] - self.params['duration']
        if value < 0:
            value = 0
        if self.params['t_start'] != value:
            self.params['t_start'] = value
            self.params['hsel_patch'].set_x(value)
            self._update_raw_data()
            self.params['plot_fun']()


    def _plot_raw_onkey(self, event):
        """Interpret key presses"""
        import matplotlib.pyplot as plt
        # check for initial plot
        if event is None:
            self.params['plot_fun']()
            return

        # quit event
        if event.key == 'escape':
            plt.close(self.params['fig'])
            return

        # change plotting self.params
        ch_changed = False
        if event.key == 'down':
            self.params['ch_start'] += self.params['n_channels']
            ch_changed = True
        elif event.key == 'up':
            self.params['ch_start'] -= self.params['n_channels']
            ch_changed = True
        elif event.key == 'right':
            self._plot_raw_time(self.params['t_start'] + self.params['duration'])
            return
        elif event.key == 'left':
            self._plot_raw_time(self.params['t_start'] - self.params['duration'])
            return
        elif event.key in ['o', 'p']:
            _toggle_options(None, self.params)
            return

        # deal with plotting changes
        if ch_changed:
            self._channels_changed()


    def _channels_changed(self):
        if self.params['ch_start'] >= len(self.params['ch_names']):
            self.params['ch_start'] = 0
        elif self.params['ch_start'] < 0:
            # wrap to end
            rem = len(self.params['ch_names']) % self.params['n_channels']
            self.params['ch_start'] = len(self.params['ch_names'])
            self.params['ch_start'] -= rem if rem != 0 else self.params['n_channels']
        self.params['plot_fun']()


    def _plot_imitate_scroll(self, start_ch):
        self.params = self.params

        orig_start = self.params['ch_start']

        self.params['ch_start'] = min( start_ch,
                                  len(self.params['ch_names']) - 
                                  self.params['n_channels'] )

        print self.params['ch_start'], 'greeib', start_ch
      
        if orig_start != self.params['ch_start']:
            #p3int self.params

            self._channels_changed()

    def _plot_raw_onscroll(self, event):
        """Interpret scroll events"""
        orig_start = self.params['ch_start']
        if event.step < 0:
            self.params['ch_start'] = min(self.params['ch_start'] + 
                                     self.params['n_channels'],
                                     len(self.params['ch_names']) -
                                     self.params['n_channels'])
        else:  # event.key == 'up':
            self.params['ch_start'] = max(self.params['ch_start'] - 
                self.params['n_channels'], 0)
        if orig_start != self.params['ch_start']:
            self._channels_changed()


    def _plot_traces(self, inds, color, bad_color, lines, event_lines,
                     event_color, offsets):
        """Helper for plotting raw"""

        #info = self.params['info']
        n_channels = self.params['n_channels']
        self.params['bad_color'] = bad_color
        # do the plotting
        tick_list = []
        for ii in range(n_channels):
            ch_ind = ii + self.params['ch_start']
            # let's be generous here and allow users to pass
            # n_channels per view >= the number of traces available
            if ii >= len(lines):
                break
            elif ch_ind < len(self.params['ch_names']):
                # scale to fit
                ch_name = self.params['ch_names'][inds[ch_ind]]
                tick_list += [ch_name]
                offset = offsets[ii]

                # do NOT operate in-place lest this get screwed up
                if inds[ch_ind] < self.params['data'].shape[0]:
                    this_data = self.params['data'][inds[ch_ind]]
                else:
                    this_data = np.zeros((1, self.params['data'].shape[1]))
                #this_color = bad_color if ch_name in info['bads'] else color
                this_color = color
                #this_z = -1 if ch_name in info['bads'] else 0
                this_z = 0
                if isinstance(this_color, dict):
                    this_color = this_color[self.params['types'][inds[ch_ind]]]

                # subtraction here gets corect orientation for flipped ylim
                # lines[ii].set_ydata(offset - this_data)
                lines[ii].set_ydata(this_data * (-1))
                lines[ii].set_xdata(self.params['times'])
                lines[ii].set_color(this_color)
                lines[ii].set_zorder(this_z)
                vars(lines[ii])['ch_name'] = ch_name
                vars(lines[ii])['def_color'] = color 
                    #color[self.params['types'][inds[ch_ind]]]
            else:
                # "remove" lines
                lines[ii].set_xdata([])
                lines[ii].set_ydata([])
        # deal with event lines
        if self.params['event_times'] is not None:
            # find events in the time window
            event_times = self.params['event_times']
            mask = np.logical_and(event_times >= self.params['times'][0],
                                  event_times <= self.params['times'][-1])
            event_times = event_times[mask]
            event_nums = self.params['event_nums'][mask]
            # plot them with appropriate colors
            # go through the list backward so we end with -1, the catchall
            used = np.zeros(len(event_times), bool)
            for ev_num, line in zip(sorted(event_color.keys())[::-1],
                                    event_lines[::-1]):
                mask = (event_nums == ev_num) if ev_num >= 0 else ~used
                assert not np.any(used[mask])
                used[mask] = True
                t = event_times[mask]
                if len(t) > 0:
                    xs = list()
                    ys = list()
                    for tt in t:
                        xs += [tt, tt, np.nan]
                        ys += [0, 2 * n_channels + 1, np.nan]
                    line.set_xdata(xs)
                    line.set_ydata(ys)
                else:
                    line.set_xdata([])
                    line.set_ydata([])
        # finalize plot
        #print self.params['times']

        # self.params['ax'].set_xlim(self.params['times'][0],
        #                       self.params['times'][0] + self.params['duration'], False)
        self.params['ax'].set_xlim(self.params['times'][0],
                              self.params['times'][0] + self.params['times'][-1], False)
        self.params['ax'].set_ylim(np.min(self.params['data']*(-1)), np.max(self.params['data']*(-1)))

        if len(tick_list) > 1:
            self.params['ax'].set_yticklabels(tick_list)
        else:
            self.params['ax'].set_title(tick_list[0])

        self.params['vsel_patch'].set_y(self.params['ch_start'])
        self.add_vline(self.params['const_event_time'])
        self.params['fig'].canvas.draw()

        #do stuff to the underlying glyph
        if self.params['glyph'] is not None:
            import color_utils 
            #restored any previously saved colors
            color_utils.change_single_glyph_color(self.params['glyph'],
                self.current_channel, self.current_channel_color)

            #set color for new channel
            self.current_channel = self.params['ch_start']
            self.current_channel_color = (
                color_utils.get_single_glyph_color( self.params['glyph'],
                    self.params['ch_start']))

            color_utils.extend_lut_with_static_color(self.params['glyph'], 
                (1, 1, .4))
            color_utils.change_single_glyph_color(self.params['glyph'],
                self.params['ch_start'], 2 )


    def add_vline(self, event_time):
        x = np.array([event_time] * 2)
        self.params['ax_vertline'].set_data(x, np.array(self.params['ax'].get_ylim()))
        self.params['ax_hscroll_vertline'].set_data(x, np.array([0., 1.]))
        self.params['vertline_t'].set_text('%0.3f' % x[0])


    def plot_raw(self, signal, events=None, duration=10.0, start=0.0, 
                 n_channels=20, bads=(),
                 bgcolor='w', color=None, bad_color=(0.8, 0.8, 0.8),
                 event_color='cyan', scalings=None, remove_dc=True, 
                 order='type', const_event_time=None,
                 show_options=False, title=None, show=False, block=False,
                 highpass=None, lowpass=None, filtorder=4, clipping=None,
                 glyph=None):

        import matplotlib.pyplot as plt
        import matplotlib as mpl

        stc = signal.mne_source_estimate

        # color, scalings = _mutable_defaults(('color', color),
        #                                     ('scalings_plot_raw', scalings))

        color = 'darkblue'

        if clipping is not None and clipping not in ('clamp', 'transparent'):
            raise ValueError('clipping must be None, "clamp", or' 
                ' "transparent" ', 
                             'not %s' % clipping)
        # figure out the IIR filtering parameters
        sfreq = 1 / stc.tstep
        nyq = sfreq / 2
        if highpass is None and lowpass is None:
            ba = None
        else:
            filtorder = int(filtorder)
            if filtorder <= 0:
                raise ValueError('filtorder (%s) must be >= 1' % filtorder)
            if highpass is not None and highpass <= 0:
                raise ValueError('highpass must be > 0, not %s' % highpass)
            if lowpass is not None and lowpass >= nyq:
                raise ValueError('lowpass must be < nyquist (%s), not %s'
                                 % (nyq, lowpass))
            if highpass is None:
                ba = butter(filtorder, lowpass / nyq, 'lowpass', analog=False)
            elif lowpass is None:
                ba = butter(filtorder, highpass / nyq, 'highpass', 
                    analog=False)
            else:
                if lowpass <= highpass:
                    raise ValueError('lowpass (%s) must be > highpass (%s)'
                                     % (lowpass, highpass))
                ba = butter(filtorder, [highpass / nyq, lowpass / nyq], 
                    'bandpass',
                            analog=False)

        # make a copy of info, remove projection (for now)
        n_times = stc.times.shape[0]

        # allow for raw objects without filename, e.g., ICA
        #if title is None:
        #    title = 'meaningless'
            #title = raw._filenames
            #if len(title) == 0:  # empty list or absent key
            #    title = '<unknown>'
            #elif len(title) == 1:
            #    title = title[0]
            #else:  # if len(title) > 1:
            #    title = '%s ... (+ %d more) ' % (title[0], len(title) - 1)
            #    if len(title) > 60:
            #        title = '...' + title[-60:]
        #elif not isinstance(title, string_types):
        #    raise TypeError('title must be None or a string')
        if events is not None:
            event_times = events[:, 0].astype(float) - stc.times[0]
            #event_times /= info['sfreq']
            event_times /= sfreq
            event_nums = events[:, 2]
        else:
            event_times = event_nums = None

        # reorganize the data in plotting order
        # do not adjust the data order at all
        inds = np.arange( stc.data.shape[0] )

        if not isinstance(event_color, dict):
            event_color = {-1: event_color}
        else:
            event_color = copy.deepcopy(event_color)  # we might modify it
        for key in event_color:
            if not isinstance(key, int):
                raise TypeError('event_color key "%s" was a %s not an int'
                                % (key, type(key)))
            if key <= 0 and key != -1:
                raise KeyError('only key <= 0 allowed is -1 (cannot use %s)'
                               % key)

        # set up projection and data parameters
        self.params = dict(stc=stc, ch_start=0, t_start=start, 
                      signal=signal,
                      glyph=glyph,
                      duration=duration,
                      remove_dc=remove_dc, ba=ba,
                      n_channels=n_channels, scalings=scalings,
                      n_times=n_times, event_times=event_times,
                      event_nums=event_nums, clipping=clipping,
                      ch_names=copy.copy(signal.ch_names),
                      projector=None, sfreq=sfreq,
                      bads=bads,const_event_time=const_event_time,
                        )

        # set up plotting
        fig = figure_nobar(facecolor=bgcolor, figsize=None)
        fig.canvas.set_window_title('mne_browse_raw')
        ax = plt.subplot2grid((10, 10), (0, 0), colspan=9, rowspan=9)
        #ax.set_title(title, fontsize=12)
        ax_hscroll = plt.subplot2grid((10, 10), (9, 0), colspan=9)
        ax_hscroll.get_yaxis().set_visible(False)
        ax_hscroll.set_xlabel('Time (s)')
        ax_vscroll = plt.subplot2grid((10, 10), (0, 9), rowspan=9)
        ax_vscroll.set_axis_off()
        ax_button = plt.subplot2grid((10, 10), (9, 9))
        # store these so they can be fixed on resize
        self.params['fig'] = fig
        self.params['ax'] = ax
        self.params['ax_hscroll'] = ax_hscroll
        self.params['ax_vscroll'] = ax_vscroll
        self.params['ax_button'] = ax_button

        # populate vertical and horizontal scrollbars
        #for ci in range(len(info['ch_names'])):
        self.params['bads'] = set(bads)
        self.params['bads'].add('EVT') # Don't show the EVT channel
        # Save the indices of the bad channels to delete them from the data afterwards
        self.params['bad_indices'] = \
            [self.params['ch_names'].index(ch_name) for
             ch_name in self.params['bads']
             if ch_name in signal.ch_names]
        self.params['ch_names_no_bads'] = [ch_name for
            ch_name in self.params['ch_names'] if
            ch_name not in self.params['bads']]
        # for bad_channel in self.params['bads']:
        #     if bad_channel in self.params['ch_names']:
        #         self.params['ch_names'].remove(bad_channel)
        for ci, ch_name in enumerate(self.params['ch_names']):
            if ch_name in self.params['bads']:
                continue
            #this_color = (bad_color if info['ch_names'][inds[ci]] in 
                #info['bads']
            #              else color)
            this_color = color
            ax_vscroll.add_patch(mpl.patches.Rectangle((0, ci), 1, 1,
                                                       facecolor=this_color,
                                                       edgecolor=this_color))
        vsel_patch = mpl.patches.Rectangle((0, 0), 1, n_channels, alpha=0.5,
                                           facecolor='w', edgecolor='w')
        ax_vscroll.add_patch(vsel_patch)
        self.params['vsel_patch'] = vsel_patch
        hsel_patch = mpl.patches.Rectangle((start, 0), duration, 1, 
            edgecolor='k',
                                           facecolor=(0.75, 0.75, 0.75),
                                           alpha=0.25, linewidth=1, 
            clip_on=False)
        ax_hscroll.add_patch(hsel_patch)
        self.params['hsel_patch'] = hsel_patch
        ax_hscroll.set_xlim(0, n_times / sfreq)
        n_ch = len(self.params['ch_names'])
        ax_vscroll.set_ylim(n_ch, 0)
        ax_vscroll.set_title('Ch.')

        # make shells for plotting traces
        offsets = np.arange(n_channels) * 2 + 1
        # plot event_line first so it's in the back
        event_lines = [ax.plot([np.nan], color=event_color[ev_num])[0]
                       for ev_num in sorted(event_color.keys())]
        ylim = [n_channels * 2 + 1, 0]

        if self.params['n_channels'] > 1:
            ax.set_yticks(offsets)
            lines = [ax.plot([np.nan])[0] for _ in range(n_ch)]
            ax.set_yticklabels(['X' * max([len(ch) for ch in self.params['ch_names']])])
            ax.set_ylim(ylim)
        else:
            lines = [ax.plot([np.nan])[0]]

        vertline_color = (0., 0.75, 0.)
        self.params['ax_vertline'] = ax.plot([0, 0], ylim, color=vertline_color,
                                        zorder=-1)[0]
        self.params['ax_vertline'].ch_name = ''
        self.params['vertline_t'] = ax_hscroll.text(0, 0.5, '0.000',
                                               color=vertline_color,
                                               verticalalignment='center',
                                               horizontalalignment='right')
        self.params['ax_hscroll_vertline'] = ax_hscroll.plot([0, 0], [0, 1],
                                                        color=vertline_color,
                                                        zorder=1)[0]

        self.params['plot_fun'] = partial(self._plot_traces,  
            inds=inds, color=color, bad_color=bad_color,
            lines=lines, event_lines=event_lines,
            event_color=event_color, offsets=offsets)

        # set up callbacks
        opt_button = mpl.widgets.Button(ax_button, 'Opt')
        callback_option = partial(_toggle_options)
        #opt_button.on_clicked(callback_option)
        callback_key = partial(self._plot_raw_onkey)
        #fig.canvas.mpl_connect('key_press_event', callback_key)
        callback_scroll = partial(self._plot_raw_onscroll)
        #fig.canvas.mpl_connect('scroll_event', callback_scroll)
        callback_pick = partial(self._mouse_click)
        #fig.canvas.mpl_connect('button_press_event', callback_pick)
        callback_resize = partial(self._helper_resize)
        #fig.canvas.mpl_connect('resize_event', callback_resize)

        # As here code is shared with plot_evoked, some extra steps:
        # first the actual plot update function

        #self.params['plot_update_proj_callback'] = _plot_update_raw_proj

        # then the toggle handler
        #callback_proj = partial(_toggle_proj, self.params=self.params)
        # store these for use by callbacks in the options figure
        #self.params['callback_proj'] = callback_proj
        self.params['callback_key'] = callback_key
        # have to store this, or it could get garbage-collected
        self.params['opt_button'] = opt_button

        # do initial plots
        #callback_proj('none')
        self._layout_raw()

        self._update_raw_data()
        self.params['plot_fun']()

        # deal with projectors
        self.params['fig_opts'] = None
        if show_options is True:
            _toggle_options(None, self.params)

        if show:
            plt.show(block=block)

        callback_dict = {'button_press_event': callback_pick,
                         'scroll_event': callback_scroll,
                         'resize_event': callback_resize,
                        }

        return fig, callback_dict, self.params

def do_browse(stc, **kwargs):
    browser = BrowseStc()
    browser.figure, cb_dict, params = browser.plot_raw(stc, **kwargs)

    browser.edit_traits()

    for cb in cb_dict:
        browser.figure.canvas.mpl_connect(cb, cb_dict[cb])

    return browser
