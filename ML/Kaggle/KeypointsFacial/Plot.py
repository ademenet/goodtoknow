import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import os


class Plot(object):
    """A bunch of methods in order to plot or save histograms based on loss"""
    def __init____(self, directory, name):
        self.train_loss, self.valid_loss, self.accuracy = np.genfromtxt(str(directory) + "/" + str(name) + ".csv", delimiter=',')

    def plot_image(x, y, path=None):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
        img = x.reshape(96, 96)
        ax.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
        ax.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)
        if path:
            plt.savefig(path)
        else:
            plt.show()

    def plot_loss_epoch(self, save=True, xlim_min=0, xlim_max=250):
        if self.train_loss is None or self.valid_loss is None:
            self.load()
        plt.plot(self.train_loss, label='Train loss')
        plt.plot(self.valid_loss, label='Validation loss')
        plt.grid()
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.xlim(xlim_min, xlim_max)
        plt.yscale("log")
        if save:
            plt.savefig(self.name)
        else:
            plt.show()


class Download(object):
    def __init__(self, name, directory=None, server='salusa'):
        self.name = name
        if not directory:
            self.directory = time.strftime('%y%m%d')
        else:
            self.directory = directory
        if server == 'server1':
            self.server = '00.000.000.000'
        elif server == 'server2':
            self.server = '00.000.000.000'

    def download(self):
        if not os.path.exists("./save/" + self.directory):
            os.makedirs("./save/" + self.directory)
        os.system("scp alain@" + str(self.server) + ":Workspace/keyface/save/" + str(self.directory) + "/" + str(self.name) + ".csv ./save/" + str(self.directory) + "/.")
        os.system("scp alain@" + str(self.server) + ":Workspace/keyface/save/" + str(self.directory) + "/" + str(self.name) + ".json ./save/" + str(self.directory) + "/.")

if len(sys.argv) == 2:
    save = Download(str(sys.argv[1]))
    save.download()
elif len(sys.argv) == 3:
    save = Download(str(sys.argv[1]), directory=str(sys.argv[2]))
    save.download()
elif len(sys.argv) == 4:
    save = Download(str(sys.argv[1]), directory=str(sys.argv[2]), server=str(sys.argv[3]))
    save.download()
else:
    print "Error!"
