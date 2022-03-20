from pypianoroll import *
from matplotlib import pyplot as plt, cm

if __name__ == '__main__':
    midifile = Multitrack("../../res/pianoroll/easy01.mid")
    midifile.plot(filename="../../out/pianoroll/easy01", label="off", cmaps=[cm.get_cmap("binary")])
    plt.show()