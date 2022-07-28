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
    TDE_OGD = [23.87, 41.91, 49.29]
    TDE = [20.35, 39.44, 46.14]
    TE = [21.24, 38.11, 46.60]
    ERR1 = [2.22, 5.88, 3.35]
    ERR2 =[1.35, 4.76, 1.33]
    ERR3 = [1.40, 3.38, 3.11]
    bar_width = 0.28
    index = np.arange(3)
    alpha = 0.6

    p1 = plt.bar(index, TE,
                 bar_width,
                 color='darkgoldenrod',
                 alpha=alpha,
                 hatch='-',
                 label='SRTD',
                 yerr=ERR3,
                 capsize=3)

    p2 = plt.bar(index + bar_width, TDE,
                 bar_width,
                 color='mediumpurple',
                 alpha=alpha,
                 label='SRTD+N',
                 yerr=ERR2,
                 capsize=3)

    p3 = plt.bar(index + bar_width * 2, TDE_OGD,
                 bar_width,
                 color='brown',
                 alpha=alpha,
                 hatch='\\',
                 label='SRTD+ID',
                 yerr=ERR3,
                 capsize=3)

    plt.legend(prop={'size': 24}, loc=(0.67, 0.03))
    plt.ylabel('Success rate (%)', fontsize=28)
    plt.xlabel('Configures', fontsize=28)
    plt.xticks([0.28, 1.28, 2.28], ['(MR 10, RP 0, ME 0)', '(MR 0, RP 10, ME 0)', '(MR 0, RP 0, ME 10)'], fontsize=22)
    plt.tight_layout(pad=1)
    plt.ylim(0, 55)
    #plt.show()
    #  plt.savefig('sim_dynamics_{}.png'.format(dyn))
    plt.savefig('sim_da.pdf')