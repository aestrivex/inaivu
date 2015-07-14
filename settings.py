from traits.api import (HasTraits, List, Float, Tuple, Instance, Bool, Str, File,
                        Int, Either, Property, Method, on_trait_change, Any, Enum, Button)
from traitsui.api import (View, Item, HGroup, Handler, CSVListEditor, VGroup,
                          InstanceEditor, Group, OKCancelButtons, TableEditor, ObjectColumn,
                          TextEditor, OKButton, CheckListEditor, Label, Action, ListStrEditor,
                          MenuBar, Menu, Tabbed, FileEditor)
from traitsui.message import error as error_dialog
from custom_list_editor import CustomListEditor

class InvasiveFile(HasTraits):

    invasive_file = File
    ordering_file = File
    file_label = Str
    source_field = Str

    traits_view = View(Item(name='invasive_file', editor=FileEditor()),
                       Item(name='ordering_file', editor=FileEditor()),
                       Item(name='file_label'),
                       Item(name='source_field'), )


class MEGFile(HasTraits):

    meg_file = File
    file_label = Str

    traits_view = View(Item(name='meg_file', editor=FileEditor()),
                       Item(name='file_label'),)


class SettingsWindow(Handler):

    model = Any
    subjects_dir = Str
    subject = Str
    montage_file = File
    invaisve_data_files = List(Instance(InvasiveFile, ())) # Not sure why, but the '()' must be here
    meg_data_files = List(Instance(MEGFile, ())) # Not sure why, but the '()' must be here
    save_values_button = Action(name = 'Save', action = '_save_values')

    traits_view = View(
        Tabbed(
            VGroup(
                Item(name='subjects_dir'),
                Item(name='subject'),
                Item(name='montage_file', editor=FileEditor()),
                label='Subject'
            ),
            VGroup(
                Label('Invaisve data files:'),
                Item('invaisve_data_files', editor=CustomListEditor(
                    editor=InstanceEditor(), style='custom', rows=5), show_label=False, ),
                label='Invaisve',
            ),
            VGroup(
                Label('MEG data files:'),
                Item('meg_data_files', editor=CustomListEditor(
                    editor=InstanceEditor(), style='custom', rows=5), show_label=False, ),
                label='MEG',
            ),
        ),
        buttons = [save_values_button],
    )

    def _save_values(self, info):
        if not info.initialized: return
        save_settings(self)
        print 'values have been saved'
        self.model.load_data(self)
        info.ui.dispose()


def load_settings():
    import json
    import os
    # from settings import InvasiveFile, MEGFile
    data = {}
    if os.path.isfile('settings.json'):
        with open('settings.json') as settings_file:
            settings = json.load(settings_file)
            # Set default values for the data files
            for k, v in settings.iteritems():
                data[k] = v
            data['invaisve_data_files'] = [InvasiveFile(
                invasive_file=f['invasive_file'],
                ordering_file=f['ordering_file'],
                file_label=f['file_label'],
                source_field=f['source_field']) for f in settings['invaisve_data_files']]
            data['meg_data_files'] = [MEGFile(
                meg_file=f['meg_file'],
                file_label=f['file_label']) for f in settings['meg_data_files']]
    if not data:
        data['invaisve_data_files'] = []
        data['meg_data_files'] = []
    return data

def save_settings(window):
    settings = {}
    # Set all the strings (unicode) properties
    for k, v in window.__dict__.iteritems():
        if isinstance(v, unicode):
            settings[k] = v
    settings['invaisve_data_files'] = []
    for f in window.invaisve_data_files:
        settings['invaisve_data_files'].append({k:v for k,v in f.__dict__.iteritems()})
    settings['meg_data_files'] = []
    for f in window.meg_data_files:
        settings['meg_data_files'].append({k:v for k,v in f.__dict__.iteritems()})
    import json
    with open('settings.json', 'w') as outfile:
        json.dump(settings, outfile)

def init_settings(model):
    settings = load_settings()
    settings_window = SettingsWindow(
        model=model,
        invaisve_data_files=settings['invaisve_data_files'],
        meg_data_files=settings['meg_data_files'],
        subjects_dir=settings['subjects_dir'],
        subject=settings['subject'],
        montage_file=settings['montage_file'])
    settings_window.edit_traits()
