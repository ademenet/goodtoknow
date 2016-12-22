import sys
import matplotlib
# matplotlib.use('GTK')
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
# import seaborn as sns
# sns.set(style="ticks", palette='GnBu_d')

# Here the basic flow to plot training and validation loss functionsb
# t_loss1, v_loss1, acc1 = np.genfromtxt("./save/161124/161124_lenet5_net8-20.csv", delimiter=',')
# t_loss2, v_loss2, acc2 = np.genfromtxt("./save/161124/161124_lenet5_net8-201.csv", delimiter=',')
# t_loss3, v_loss3, acc3 = np.genfromtxt("./save/161124/161124_lenet5_net8-2012.csv", delimiter=',')
# t_loss4, v_loss4, acc4 = np.genfromtxt("./save/161124/161124_lenet5_net8-20123.csv", delimiter=',')
# t_loss5, v_loss5, acc5 = np.genfromtxt("./save/161124/161124_lenet5_net8-201234.csv", delimiter=',')
# t_loss6, v_loss6, acc6 = np.genfromtxt("./save/161124/161124_lenet5_net8-2012345.csv", delimiter=',')
# t_loss7, v_loss7, acc7 = np.genfromtxt("./save/161124/161124_lenet5_net8-20123456.csv", delimiter=',')
# t_loss8, v_loss8, acc8 = np.genfromtxt("./save/161124/161124_lenet5_net8-201234567.csv", delimiter=',')
# t_loss9, v_loss9, acc9 = np.genfromtxt("./save/161124/161124_lenet5_net8-2012345678.csv", delimiter=',')
# t_loss10, v_loss10, acc10 = np.genfromtxt("./save/161124/161124_lenet5_net8-20123456789.csv", delimiter=',')
# t_loss11, v_loss11, acc11 = np.genfromtxt("./save/161124/161124_lenet5_net8-2012345678910.csv", delimiter=',')
# t_loss12, v_loss12, acc12 = np.genfromtxt("./save/161124/161124_lenet5_net8-201234567891011.csv", delimiter=',')

t_loss1, v_loss1, acc1 = np.genfromtxt("./save/161124/161124_lenet5_net8-3s.csv", delimiter=',')
t_loss2, v_loss2, acc2 = np.genfromtxt("./save/161201/161201_lenet5_net10-1s-eyes.csv", delimiter=',')
t_loss3, v_loss3, acc3 = np.genfromtxt("./save/161201/161201_lenet5_net10-1s-nose.csv", delimiter=',')
t_loss4, v_loss4, acc4 = np.genfromtxt("./save/161202/161202_lenet5_net10-1s-noseNmouth.csv", delimiter=',')
# t_loss5, v_loss5, acc5 = np.genfromtxt("./save/161130/161130_BigConv2_net9-BC2s.csv", delimiter=',')
# t_loss6, v_loss6, acc6 = np.genfromtxt("./save/161130/161130_BigConv_net9-BC1s.csv", delimiter=',')

# t_loss1, v_loss1, acc1 = np.genfromtxt("./save/161124/161124_lenet5_net8-3t.csv", delimiter=',')
# t_loss2, v_loss2, acc2 = np.genfromtxt("./save/161125/161125_lenet5_net8-3t2.csv", delimiter=',')
# t_loss3, v_loss3, acc3 = np.genfromtxt("./save/161124/161124_lenet5_net8-3s.csv", delimiter=',')
# t_loss4, v_loss4, acc4 = np.genfromtxt("./save/161122/161122_lenet5_net8-1s5.csv", delimiter=',')
# t_loss5, v_loss5, acc5 = np.genfromtxt("./save/161125/161125_model1_net9-0t.csv", delimiter=',')
# t_loss7, v_loss7, acc7 = np.genfromtxt("./save/161122/161122_lenet5_net8-1s6.csv", delimiter=',')
# t_loss8, v_loss8, acc8 = np.genfromtxt("./save/161121/161121_lenet5_net7-1s.csv", delimiter=',')

c = {
	'red': '#F44336',
	'deeppurple': '#673AB7',
	'blue': '#2196F3',
	'cyan': '#00BCD4',
	'lime': '#CDDC39',
	'orange': '#FF9800',
	'yellow': '#FFEB3B',
	'brown': '#795548'
}

g = {
	'g_bg1': '#B2EBF2',
	'g_bg2': '#80DEEA',
	'g_bg3': '#4DD0E1',
	'g_bg4': '#26C6DA',
	'g_bg5': '#00BCD4',
	'g_bg6': '#00ACC1',
	'g_bg7': '#0097A7',
	'g_bg8': '#00838F'
}

mpl.rcParams['lines.linewidth'] = 2
plt.figure(figsize=(14, 10))

plt.plot(v_loss1, label='net8-3s (v)', color=c['red'])
plt.plot(t_loss1, label='net8-3s (t)', linestyle='--', color=c['red'])

plt.plot(v_loss2, label='net10-1s-eyes (v)', color=c['cyan'])
plt.plot(t_loss2, label='net10-1s-eyes (t)', linestyle='--', color=c['cyan'])

plt.plot(v_loss3, label='net10-1s-nose (v)', color=c['blue'])
plt.plot(t_loss3, label='net10-1s-nose (t)', linestyle='--', color=c['blue'])

plt.plot(v_loss4, label='net10-1s-noseNmouth (v)', color=c['lime'])
plt.plot(t_loss4, label='net10-1s-noseNmouth (t)', linestyle='--', color=c['lime'])

# plt.plot(v_loss5, label='net9-BC2s (v)', color=c['orange'])
# plt.plot(t_loss5, label='net9-BC2s (t)', linestyle='--', color=c['orange'])

# plt.plot(v_loss6, label='net9-BC1s (v)', color=c['brown'])
# plt.plot(t_loss6, label='net9-BC1s (t)', linestyle='--', color=c['brown'])

# plt.plot(v_loss7, label='net8-3t (v)')
# # plt.plot(t_loss7, label='net8-3t (t)', linestyle='--')

# plt.plot(v_loss8, label='net8-3t (v)')
# # plt.plot(t_loss8, label='net8-3t (t)', linestyle='--')

# plt.plot(v_loss9, label='net8-3t (v)')
# # plt.plot(t_loss9, label='net8-3t (t)', linestyle='--')

# plt.plot(v_loss10, label='net8-3t (v)')
# # plt.plot(t_loss10, label='net8-3t (t)', linestyle='--')

# plt.plot(v_loss11, label='net8-3t (v)')
# # plt.plot(t_loss11, label='net8-3t (t)', linestyle='--')

# plt.plot(v_loss12, label='net8-3t (v)')
# # plt.plot(t_loss12, label='net8-3t (t)', linestyle='--')

# plt.plot(v_loss2, label='net9-0s (v)', color=c['blue'])
# plt.plot(t_loss2, label='net9-0s (t)', linestyle='--', color=c['blue'])

# plt.plot(v_loss3, label='1e-2 net8-1s2 (v)', color=g['g_bg3'])
# plt.plot(t_loss3, label='1e-2 net8-1s2 (t)', linestyle='--', color=g['g_bg3'])

# plt.plot(v_loss4, label='1e-3 net8-1s3 (v)', color=g['g_bg4'])
# plt.plot(t_loss4, label='1e-3 net8-1s3 (t)', linestyle='--', color=g['g_bg4'])

# plt.plot(v_loss5, label='1e-4 net8-1s4 (v)', color=g['g_bg5'])
# plt.plot(t_loss5, label='1e-4 net8-1s4 (t)', linestyle='--', color=g['g_bg5'])

# plt.plot(v_loss6, label='1e-5 net8-1s5 (v)', color=g['g_bg6'])
# plt.plot(t_loss6, label='1e-5 net8-1s5 (t)', linestyle='--', color=g['g_bg6'])

# plt.plot(v_loss7, label='1e-6 net8-1s6 (v)', color=g['g_bg7'])
# plt.plot(t_loss7, label='1e-6 net8-1s6 (t)', linestyle='--', color=g['g_bg7'])

# plt.plot(v_loss8, label='0 net7-1s (v)', color=c['lime'])
# plt.plot(t_loss8, label='0 net7-1s (v)', linestyle='--', color=c['lime'])

# plt.plot(t_loss5, label='Glorot Uniform (default)', color='y')
# plt.plot(v_loss5, color='y', linestyle='--')

# Accuracy

# plt.plot(acc1, label='net7-2a1')
# plt.plot(acc2, label='net7-2a2')
# plt.plot(acc3, label='net7-2a3')
# plt.plot(acc4, label='net7-2a4')
# plt.plot(acc5, label='net7-1')

title('net8-3')
plt.grid()
plt.legend(fontsize='x-small')
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.xlim(-5, 2000)
# plt.ylim(1e-3, 1e0)
plt.yscale("log")
# sns.despine()

yes = set(['yes','y', 'ye', ''])
if len(sys.argv) == 2:
	print "Save as " + str(sys.argv[1]) + '.png?'
	choice = raw_input().lower()
	if choice in yes:
		plt.savefig('doc/img/' + str(sys.argv[1]) + '.png')
	else:
		sys.exit()
else:
	plt.show()
