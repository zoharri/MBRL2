# from packaging import version
#
# import pandas as pd
# from matplotlib import pyplot as plt
# import seaborn as sns
# from scipy import stats
# import tensorboard as tb
#
#
# experiment_id = "c1KCv3X3QvGwaXfgX1c4tg"
# experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
# df = experiment.get_scalars()
# print(df.head(10))
#
# from tensorboard.backend.event_processing import event_accumulator
# import glob
# import pandas as pd
# import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def get_tb_data(tb_path, scalar_key='eval_return_avg_per_frame/sum_episodes'):
    event_acc = EventAccumulator(tb_path)
    event_acc.Reload()

    # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
    _, step_nums, vals = zip(*event_acc.Scalars(scalar_key))
    return step_nums, vals
