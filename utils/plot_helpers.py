from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm


def get_tb_data(tb_path, scalar_key='eval_return_avg_per_frame/sum_episodes'):
    event_acc = EventAccumulator(tb_path)
    event_acc.Reload()

    # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
    _, step_nums, vals = zip(*event_acc.Scalars(scalar_key))
    return step_nums, vals


def get_dir(base_path: Path, method_name: str, seed: int):
    prefix = method_name + "_" + str(seed) + "_"
    prefixed = [filename for filename in os.listdir(base_path) if filename.startswith(prefix)]
    return str(base_path / prefixed[0])  # Hopefully we don't have more than one


sns.set(style="darkgrid")

# --------------------------------------------------------------------
# SETTINGS
# --------------------------------------------------------------------


# which methods to plot

# Half 20
envs = ['HalfCircle']
methods = ['with_dream_4_dreams_20_empkde_scratch_update3_delay5000',
           'with_dream_4_dreams_20_mixup_empkde_scratch_update3_delay5000',
           'no_dream_20_vae_training_from_ldm_old_vae_params',
           ]
fill_between = [True, False, True]
seeds = [[3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123, 133, 143],
         [3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123, 133, 143],
         [3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 200, 201, 202, 203],
         ]

title_name = "HalfCircle - 20 Real Environments"
max_len = 428

# Half 30
# envs = ['HalfCircle']
# methods = ['with_dream_6_dreams_30_empkde_scratch_update3_delay5000',
#            'with_dream_6_dreams_30_mixup_empkde_scratch_update3_delay5000',
#            'no_dream_30_vae_training_from_ldm_old_vae_params']


# seeds = [[3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 200, 201, 202, 203],
#          [3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 200, 201, 202, 203],
#          [3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 200, 201, 202, 203]]
# title_name = "HalfCircle - 30 Real Environments"
# fill_between = [True, False, True]
# max_len = 428

# Full 30
# envs = ['FullCircle']
# methods = ['with_dream_2_dreams_30_empkde_scratch_update3_delay5000',
#            'no_dream_30_vae_training_from_ldm_old_vae_params']
# seeds = [[3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123, 133, 143, 153, 163, 173, 183, 193],
#          [3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123, 133, 143, 153, 163, 173, 183, 193]]
# max_len = 460
# title_name = "FullCircle - 30 Real Environments"

# Full 40
# envs = ['FullCircle']
# methods = ['with_dream_2_dreams_40_empkde_scratch_update3_delay5000',
#            'no_dream_40_vae_training_from_ldm_old_vae_params']
# seeds = [[3, 13, 23, 33, 43, 53, 63, 73, 93, 103, 113, 123],
#          [3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123, 133, 143]]
# title_name = "FullCircle - 40 Real Environments"


# which environments to plot

base_paths = {'HalfCircle': Path('/path/to/proj/MBRL2/logs/logs_SparsePointEnv-v0'),
              'FullCircle': Path('/path/to/proj/MBRL2/logs/logs_SparsePointEnvFull-v0')}

# whether you want the smoothened curves or not
smoothing_factor = 10  # 0 to disable
# --------------------------------------------------------------------
# PLOTTING SCRIPT
# --------------------------------------------------------------------

# some settings for plotting
my_colors = {
    'with_dream_6_dreams_30_empkde_scratch_update3_delay5000': sns.color_palette("bright", 10)[3],
    'with_dream_4_dreams_20_mixup_empkde_scratch_update3_delay5000': sns.color_palette("bright", 10)[2],
    'with_dream_6_dreams_30_mixup_empkde_scratch_update3_delay5000': sns.color_palette("bright", 10)[2],
    'with_dream_2_dreams_30_empkde_scratch_update3_delay5000': sns.color_palette("bright", 10)[3],
    'with_dream_2_dreams_40_empkde_scratch_update3_delay5000': sns.color_palette("bright", 10)[3],
    'with_dream_4_dreams_20_empkde_scratch_update3_delay5000': sns.color_palette("bright", 10)[3],
    'with_dream_4_dreams_40_empkde_scratch_update3_delay5000': sns.color_palette("bright", 10)[3],
    'with_dream_6_dreams_20_empkde_scratch_update3_delay5000': sns.color_palette("bright", 10)[3],
    'no_dream_10_vae_training_from_ldm_old_vae_params': sns.color_palette("bright", 10)[7],
    'no_dream_20_vae_training_from_ldm_old_vae_params': sns.color_palette("bright", 10)[7],
    'no_dream_30_vae_training_from_ldm_old_vae_params': sns.color_palette("bright", 10)[7],
    'no_dream_40_vae_training_from_ldm_old_vae_params': sns.color_palette("bright", 10)[7],

}

my_labels = {
    'with_dream_6_dreams_30_empkde_scratch_update3_delayfreeze12500': 'w/dream envs (Ours)',
    'with_dream_6_dreams_30_empkde_scratch_update3_delay5000': 'w/ dream envs (Ours)',
    'with_dream_4_dreams_20_mixup_empkde_scratch_update3_delay5000': 'w/ mixup dream envs',
    'with_dream_6_dreams_30_mixup_empkde_scratch_update3_delay5000': 'w/ mixup dream envs',
    'with_dream_2_dreams_30_empkde_scratch_update3_delay5000': 'w/ dream envs (Ours)',
    'with_dream_2_dreams_40_empkde_scratch_update3_delay5000': 'w/ dream envs (Ours)',
    'with_dream_4_dreams_10_trained_vae_update50_4999_policydream_from_ldm_old_vae': 'w/ dream envs (Ours)',
    'with_dream_4_dreams_40_empkde_scratch_update3_delay5000': 'w/ dream envs (Ours)',
    'no_dream_40_vae_training_from_ldm_old_vae_params': 'w/ dream envs (Ours)',
    'with_dream_4_dreams_20_empkde_scratch_update3_delay5000': 'w/ dream envs (Ours)',
    'with_dream_6_dreams_20_empkde_scratch_update3_delay5000': 'w/ dream envs (Ours)',
    'no_dream_10_vae_training_from_ldm_old_vae_params': 'w/o dream envs',
    'no_dream_20_vae_training_from_ldm_old_vae_params': 'w/o dream envs',
    'no_dream_30_vae_training_from_ldm_old_vae_params': 'w/o dream envs',
    'with_dream_6_dreams_20_trained_vae_scrach_update50_policydream_from_ldm_old_vae': 'w/ dream envs (Ours)',
    'with_dream_6_dreams_30_trained_vae_scrach_update50_policydream_from_ldm_old_vae': 'w/ dream envs (Ours)',
    'with_dream_10_trained_vae_update50_4999_policydream_from_ldm_old_vae': 'w/ dream envs (Ours)',
    'with_dream_2_dreams_10_trained_vae_update50_4999_policydream_from_ldm_old_vae': 'w/ dream envs (Ours)',
    'with_dream_4_dreams_10_trained_vae_update1_4999_policydream_from_ldm_old_vae': 'w/ dream envs (Ours)',
}

my_linestyles = {
    'with_dream_4_dreams_10_trained_vae_update50_4999_policydream_from_ldm_old_vae': '--',
    'with_dream_6_dreams_30_empkde_scratch_update3_delay5000': '--',
    'with_dream_4_dreams_40_empkde_scratch_update3_delay5000': '--',
    'with_dream_2_dreams_30_empkde_scratch_update3_delay5000': '--',
    'with_dream_2_dreams_40_empkde_scratch_update3_delay5000': '--',
    'with_dream_4_dreams_20_mixup_empkde_scratch_update3_delay5000': '--',
    'with_dream_6_dreams_30_mixup_empkde_scratch_update3_delay5000': '--',
    'no_dream_10_vae_training_from_ldm_old_vae_params': '--',
    'no_dream_40_vae_training_from_ldm_old_vae_params': '--',
    'with_dream_4_dreams_20_empkde_scratch_update3_delay5000': '--',
    'with_dream_6_dreams_20_empkde_scratch_update3_delay5000': '--',
    'with_dream_10_trained_vae_update50_4999_policydream_from_ldm_old_vae': '--',
    'no_dream_20_vae_training_from_ldm_old_vae_params': '--',
    'no_dream_30_vae_training_from_ldm_old_vae_params': '--',
    'with_dream_6_dreams_20_trained_vae_scrach_update50_policydream_from_ldm_old_vae': '--',
    'with_dream_6_dreams_30_empkde_scratch_update3_delayfreeze12500': '--',
    'with_dream_6_dreams_30_trained_vae_scrach_update50_policydream_from_ldm_old_vae': '--',
    'with_dream_2_dreams_10_trained_vae_update50_4999_policydream_from_ldm_old_vae': '--',
    'with_dream_4_dreams_10_trained_vae_update1_4999_policydream_from_ldm_old_vae': '--',
}

plot_all_seeds = False

# plot results for each env and method

for env in envs:
    plt.figure(figsize=(8, 6))
    for i, method in enumerate(methods):
        print(f"At method: {method}")
        x = []
        y = []
        curr_seeds = seeds[i] if isinstance(seeds[0], list) else seeds

        for seed in tqdm(curr_seeds):
            seed_x, seed_y = get_tb_data(get_dir(base_paths[env], method, seed))
            seed_x, seed_y = seed_x[:max_len], seed_y[:max_len]
            x.append(np.array(seed_x))

            if smoothing_factor != 0:
                seed_y = np.convolve(seed_y, np.ones(smoothing_factor) / smoothing_factor, mode='valid')
                y.append(seed_y)
            else:
                y.append(np.array(seed_y))
            if plot_all_seeds is True:
                plt.plot(x[0][:-smoothing_factor + 1] if smoothing_factor is not 0 else x[0], seed_y,
                         linestyle=my_linestyles[method], linewidth=0.5,
                         color=my_colors[method])
        x = np.array(x[0][:-smoothing_factor + 1] if smoothing_factor is not 0 else x[0])
        y = np.array(y)
        # compute averages and confidence intervals across seeds
        y_mean = np.mean(y, axis=0)
        y_std = np.std(y, axis=0)
        y_se = scipy.stats.sem(y, axis=0)
        y_cfi = (np.percentile(y, 25, axis=0), np.percentile(y, 75, axis=0))

        # plot
        p = plt.plot(x, y_mean, linestyle=my_linestyles[method], linewidth=2, label=my_labels[method],
                     color=my_colors[method])
        if fill_between[i] is True:
            plt.gca().fill_between(x, y_cfi[0], y_cfi[1], facecolor=p[0].get_color(), alpha=0.4)
        print(f"Mean: {y_mean[-120:-35].mean()}")
        print(f"Mean std: {y_std[-120:-35].mean()}")
        print(len(y_mean))

    plt.title(title_name, fontsize=20)
    plt.xlabel('Frames', fontsize=20)
    plt.ylabel('Average Eval Return', fontsize=20)
    plt.gca().tick_params(axis='both', which='minor', labelsize=20)
    plt.ylim([20, 160])
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
