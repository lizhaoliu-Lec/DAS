import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# def main():
#     frm1 = np.loadtxt('frm/frm-epoch-299.txt')
#     frm2 = np.loadtxt('frm/frm-epoch-150.txt')
#     frms = [frm1, frm2]
#     print("===> frm.shape: {}".format(frm1.shape))
#     print("===> frm {}".format(frm1))
#
#     fig, axn = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(12, 6))
#     plt.axis('off')
#     # cbar_ax = fig.add_axes([.91, .3, .03, .4])
#     cbar_ax = fig.add_axes([.91, .3, .03, .4])
#
#     for i, (ax, frm) in enumerate(zip(axn.flat, frms)):
#         ax.axis('off')
#         sns.heatmap(frm[:49, :], ax=ax,
#                     cbar=i == 0,
#                     vmin=0, vmax=1,
#                     cbar_ax=None if i else cbar_ax)
#         # plt.xlabel('channel index')
#         # plt.ylabel('class index')
#
#     fig.tight_layout(rect=[0, 0, .9, 1])
#     plt.show()
#     plt.close()
# ValueError: Colormap flare is not recognized. Possible values are: Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, icefire, icefire_r, inferno, inferno_r, jet, jet_r, magma, magma_r, mako, mako_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, rocket, rocket_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, twilight, twilight_r, twilight_shifted, twilight_shifted_r, viridis, viridis_r, vlag, vlag_r, winter, winter_r
import tqdm


def main_separate():
    DATASETS = ['car', 'cub', 'sop']
    COLOR_MAPS = ['jet', 'Blues']
    FONT_SIZE = 30
    for dataset in DATASETS:
        print("===> Plotting for dataset: {}".format(dataset))
        # print("===> plt.colormaps(): {}".format(plt.colormaps()))
        frm2 = np.loadtxt('frm/{}/frm-epoch-299.txt'.format(dataset))

        # for colormap in tqdm.tqdm(plt.colormaps()):
        for colormap in tqdm.tqdm(COLOR_MAPS):
            plt.figure(figsize=(12, 6))
            ax = sns.heatmap(frm2[:49, :], vmin=0, vmax=1, cmap=colormap, cbar=False)
            cbar = ax.collections[0].colorbar
            # here set the labelsize by 20
            cbar.ax.tick_params(labelsize=FONT_SIZE)
            plt.xlabel('Channel Index', fontsize=FONT_SIZE)
            plt.ylabel('Class Index', fontsize=FONT_SIZE)
            plt.xticks([])
            plt.yticks([])
            # plt.axis('off')
            plt.tight_layout()
            # plt.show()
            save_path = 'final_ret/{}'.format(dataset)
            # create vis path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, '{}-{}.pdf'.format(dataset, colormap)),
                        dpi=300,
                        bbox_inches='tight',
                        pad_inches=0.0)
            plt.close()


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns


def get_alpha_blend_cmap(cmap, alpha):
    cls = plt.get_cmap(cmap)(np.linspace(0, 1, 256))
    cls = (1 - alpha) + alpha * cls
    return ListedColormap(cls)


def main_combine():
    NUM_EPOCH = 1
    # NUM_EPOCH = 149
    # NUM_EPOCH = 299
    COLOR_MAP = 'jet'
    TRANSPARENCY = 0.9
    DATASETS = ['car', 'cub', 'sop']
    TITLES = ['CARS', 'CUB', 'SOP']
    FRMS = [np.loadtxt('frm/{}/frm-epoch-{}.txt'.format(dataset, NUM_EPOCH)) for dataset in DATASETS]
    FONT_SIZE = 35
    fig, axn = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(40, 9))
    fig.tight_layout()
    # cbar_ax = fig.add_axes([.91, .3, .03, .4])
    cbar_ax = fig.add_axes([.91, .12, .01, .8])

    for i, (ax, frm) in enumerate(zip(axn.flat, FRMS)):
        # ax.set_figure(figsize=(12, 6))
        sns.heatmap(frm[:48, :], ax=ax,
                    cbar=i == 0,
                    vmin=0, vmax=1,
                    cbar_ax=None if i else cbar_ax,
                    cmap=COLOR_MAP, linewidths=0.0)

        plt.subplots_adjust(wspace=10.0)

        cbar = ax.collections[0].colorbar
        if cbar is not None:
            # here set the labelsize by 20
            cbar.ax.tick_params(labelsize=FONT_SIZE)
        # ax.axis["bottom"].set_axisline_style("-|>", size=1.5)
        # ax.axis["left"].set_axisline_style("->", size=1.5)
        ax.set_xlabel('Channel Index', fontsize=FONT_SIZE)
        ax.set_ylabel('Class Index', fontsize=FONT_SIZE)
        ax.set_title('{}'.format(TITLES[i]), fontsize=FONT_SIZE)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(rect=[0, 0, .9, 1])
    # plt.show()
    plt.savefig(os.path.join('final_ret',
                             'epoch-{}-combined-{}-titled.pdf'.format(NUM_EPOCH, COLOR_MAP)),
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.0)
    plt.close()


if __name__ == '__main__':
    # main_separate()
    main_combine()
