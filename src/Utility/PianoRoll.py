from pypianoroll import *
from matplotlib import pyplot as plt, cm

midifile = Multitrack("../../res/pianoroll/merged.mid")
midifile.to_pretty_midi()

midifile.plot(filename="../../out/pianoroll/merged", label="off", cmaps=[cm.get_cmap("binary")])
plt.show()

