#    (C) Roan LaPlante 2013 rlaplant@nmr.mgh.harvard.edu
#
#	 This file is part of cvu, the Connectome Visualization Utility.
#
#    cvu is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

from traits.trait_base import ETSConfig
_tk = ETSConfig.toolkit

if (_tk is None) or (_tk == 'null'):
    raise NotImplementedError("We must independently set the toolkit")

from traits.api import Any, Int, Bool, Instance, Either, Dict
from traitsui.basic_editor_factory import BasicEditorFactory
from matplotlib.figure import Figure


#import wx
#from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
#from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
# this import is not portable. But that is ok because currently the number of options
# is exactly 2
FigureCanvas = getattr(__import__('matplotlib.backends.backend_%sagg'%_tk,
    fromlist=['FigureCanvas']), 'FigureCanvas%sAgg'%('Wx' if _tk=='wx' else 'QT'))

#from traitsui.<toolkit>.editor import Editor
Editor = __import__('traitsui.%s.editor'%_tk,fromlist=['Editor']).Editor

import numpy as np
import time 

#This code is extensively adapted from Gael Varoquax's example code for
#hacking a traited matplotlib editor

class _MPLFigureEditor(Editor):

    scrollable = True
    parent = Any
    canvas = Instance(FigureCanvas)

    def init(self,parent):
        self.parent=parent
        self.control=self._create_canvas(parent)

    def update_editor(self):
        pass

    def _create_canvas(self, *args):
        return getattr(self,'_create_canvas_%s'%_tk)(*args)

    def _create_canvas_wx(self, parent):
        import wx
        #unsure if there is a way to avoid hard coding these function names
        fig=self.object.figure
        panel=wx.Panel(parent,-1)
        self.canvas = canvas = FigureCanvas(panel,-1,fig)
        sizer=wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas,1,wx.EXPAND|wx.ALL,1)
        #toolbar=NavigationToolbar2Wx(self.canvas)
        #toolbar.Realize()
        #sizer.Add(toolbar,0,wx.EXPAND|wx.ALL,1)
        panel.SetSizer(sizer)

        #for the panning process the id of the callback must be stored
        #self.motion_cid=self.canvas.mpl_connect('motion_notify_event',
        #	lambda ev:self.object.circ_mouseover(ev,self))

        #canvas.mpl_connect('button_press_event',self.object.circle_click)
        #canvas.mpl_connect('motion_notify_event',

        for cb in cb_dict:
            canvas.mpl_connect(cb, cb_dict[cb])

        #self.tooltip=wx.ToolTip(tip='')
        #self.tooltip.SetDelay(2000)
        #canvas.SetToolTip(self.tooltip)
        return panel

    def _create_canvas_qt4(self, parent):
        import matplotlib
        #matplotlib.use('Qt4Agg')
        #matplotlib.rcParams['backend.qt4']='PySide'

        from pyface.qt import QtCore, QtGui

        #self.tooltip = panel = QtGui.QWidget()
        panel = QtGui.QWidget()

        fig = self.object.figure
        self.canvas = canvas = FigureCanvas(fig)
        #self.canvas.setParent(panel)

        layout = QtGui.QVBoxLayout( panel )
        layout.addWidget(canvas)

        return panel

class MPLFigureEditor(BasicEditorFactory):
    klass = _MPLFigureEditor
