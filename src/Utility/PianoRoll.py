from pypianoroll import *
from matplotlib import pyplot as plt, cm

midifile = Multitrack("../../res/pianoroll/medium02.mid")
midifile.plot(filename="../../out/pianoroll/medium02", label="off", cmaps=[cm.get_cmap("binary")])
plt.show()

