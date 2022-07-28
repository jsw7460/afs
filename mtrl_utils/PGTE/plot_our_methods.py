import matplotlib.pyplot as plt
import pickle as pkl
import copy
import numpy as np
plt.rcParams['font.family'] = 'Times new roman'
plt.rcParams['font.size'] = 20
plt.rcParams["figure.figsize"] = (10, 6)

def make_plot(_data):
    _data2_smooth = np.zeros([len(_data), 21])
    _data2_smooth_x = np.arange(21) * 10000
    for i in range(len(_data)):
        min_value = 0
        max_value = 0
        z = 1
        for k in range(len(_data[i][0])):
            if _data[i][0][k] > 10000 * z:
                max_value = k
                _data2_smooth[i][z] += np.average(_data[i][1][min_value: max_value])
                min_value = k
                z += 1
            if z > 20:
                break
    for j in range(len(_data)):
        for i in range(21):
            _data2_smooth[j][i] = np.mean(_data2_smooth[j][max(i - 1, 0): min(i + 1, 20)])
    print(_data2_smooth)
    _data2_smooth_y = np.mean(_data2_smooth, axis=0)
    _data2_smooth_std = np.std(_data2_smooth, axis=0)
    return _data2_smooth_x, _data2_smooth_y, _data2_smooth_std

if __name__ == '__main__':
    TDE_OGD = [20.09, 33.38, 39.93]
    TDE = [18.03, 28.04, 38.93]
    TE = [21.24, 38.11, 46.60]
    ERR1 = [2.08, 4.55, 1.68]
    ERR2 =[1.51, 2.56, 3.19]
    ERR3 = [1.40, 3.38, 3.11]
    bar_width = 0.28
    index = np.arange(3)
    alpha = 0.6
    p1 = plt.bar(index, TDE_OGD,
                 bar_width,
                 color='mediumseagreen',
                 alpha=1,
                 label='TE',
                 yerr=ERR1,
                 capsize=3)

    p2 = plt.bar(index + bar_width, TDE,
                 bar_width,
                 color='deepskyblue',
                 alpha=alpha,
                 hatch='/',
                 label='SRTD-Q',
                 yerr=ERR2,
                 capsize=3)

    p3 = plt.bar(index + bar_width * 2, TE,
                 bar_width,
                 color='darkgoldenrod',
                 alpha=alpha,
                 hatch='-',
                 label='SRTD',
                 yerr=ERR3,
                 capsize=3)

    plt.legend(prop={'size': 24}, loc=(0.7, 0.03))
    plt.ylabel('Success rate (%)', fontsize=28)
    plt.xlabel('Configures', fontsize=28)
    plt.xticks([0.28, 1.28, 2.28], ['(MR 10, RP 0, ME 0)', '(MR 0, RP 10, ME 0)', '(MR 0, RP 0, ME 10)'], fontsize=22)
    plt.tight_layout(pad=1)
    plt.ylim(0, 55)
    #plt.show()
    #  plt.savefig('sim_dynamics_{}.png'.format(dyn))
    plt.savefig('sim_oursol.pdf')