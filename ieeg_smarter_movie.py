import inaivu

subjects_dir = '/cluster/neuromind/rlaplant/ct/mg51'
subject = 'fake_subject'

im = inaivu.InaivuModel()

brain, figure = im.build_surface(subject=subject, subjects_dir=subjects_dir)

raw = '/space/truffles/2/users/rlaplant/ecog/converted_gr.fif'

ieeg_glyph = im.plot_ieeg( raw, figure=figure )

#inaivu.show()
import timesignal

#invsig = timesignal.gen_random_invasive_signal(im.ch_names, 100)
invsig = timesignal.gen_stupid_gamma_signal(im.ch_names)

im.add_invasive_signal(1, invsig)

im.movie('test_ieeg_movie.mov', noninvasive=False)
