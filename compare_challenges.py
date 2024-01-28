import json

import os

from main import get_architecture_length
import numpy as np
import pandas as pd
root, _, files = next(os.walk('data/SwitchableStarPUF'))

challenges = []
for file in files:
    if not "64" in file:
        continue
    length = get_architecture_length(file)
    ids = list(range(length))
    np.random.shuffle(ids)
    data_dir = f'{root}/{file}'

    cbits = int(file.split('bit')[0])


    data_file = file
    # data.columns.get_loc('2.034068136272545e-08') -> 203
    timestamp = 75
    data = pd.read_csv(data_dir).iloc[ids]
    data.columns = ['challenge'] + list(
        range(1, len(data.columns)))
    data = data[['challenge', timestamp]]
    data = data.sample(frac=1, random_state=17)
    challenges.append(data['challenge'].values)
    if len(challenges) > 1:
        print(set(challenges[0]).intersection(set(challenges[1])))
        exit()


