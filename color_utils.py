import numpy as np

def extend_lut_with_static_color(mayavi_obj, color, use_vector_lut=False):
    '''
    sets the lookup table of the mayavi object provided to a 512 color map.
    expects a 256 color map (true of default maps). Sets data range to [0,2].
    Values between [0,1] function normally. Values between (1,2) have
    undefined behavior. Values of [2] show color specified in argument.
        
    
    mayavi_obj : Instance(mlab.pipeline)
        A mayavi object with a module_manager instance
    color : 3-tuple
        An RGB color represented as a 3-tuple with values [0,1], and
        corresponding to the scalar value 0
    '''
    mayavi_obj.actor.mapper.scalar_visibility = True

    if use_vector_lut:
        lut_mgr=mayavi_obj.module_manager.vector_lut_manager
    else:
        lut_mgr=mayavi_obj.module_manager.scalar_lut_manager

    if len(lut_mgr.lut.table) != 512:
        new_table = np.zeros((512, 4))
        new_table[:256] = lut_mgr.lut.table
        new_table[-1] = [c*255 for c in color] + [255]
        lut_mgr.lut.table = new_table
    lut_mgr.data_range = [0,2]

def get_single_glyph_color(mayavi_glyph, index):
    '''
    gets the scalar value at a particular index of a glyph
    '''
    colors = np.array(mayavi_glyph.mlab_source.dataset.point_data.scalars)
    return colors[index]

def change_single_glyph_color(mayavi_glyph, index, color):
    '''
    changes the scalars trait of a mayavi glyph safely to display the altered
    color of a single node

    Parameters
    ----------

    mayavi_glyph : mayavi.modules.Glyph
        The glyph object
    index : int
        The offset of the node within the glyph
    color : number
        The new value of the scalar to set at this index
    '''
    colors = np.array(mayavi_glyph.mlab_source.dataset.point_data.scalars)
    colors[index] = color
    mayavi_glyph.mlab_source.dataset.point_data.scalars = colors
