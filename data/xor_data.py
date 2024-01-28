import os

import pandas as pd


def prepare_data_file(data_new, idx):
    data_new.columns = ['challenge'] + list(range(1, len(data_new.columns)))
    data_new = data_new[['challenge', timestamp]]
    data_new.columns = ['challenge', idx]
    data_new[f'msb{idx}'] = data_new[idx] >= 10
    data_new[f'lsb{idx}'] = data_new[idx] % 2 == 1
    return data_new

folder = 'SwitchableStarPUF'
_, _, files = next(os.walk(folder))
n = '32'
timestamp = 7
sep_type = '0_sep'

files_nbit = [x for x in files if n in x]
files_nbit_sep = [x for x in files_nbit if sep_type in x]

data_start = pd.read_csv(f'{folder}/{files_nbit_sep[0]}')
data_start = prepare_data_file(data_start, 0)
data_iterator = []
for idx, file_new in enumerate(files_nbit_sep[1:], 1):
    data_new = pd.read_csv(f'{folder}/{file_new}')
    data_new = prepare_data_file(data_new, idx)
    data_iterator.append(data_new)
    data_m = data_start
    for idx2, data_p in enumerate(data_iterator, 1):
        data_m = data_m.merge(data_p, on='challenge')
        data_m['msb0'] = data_m['msb0'] ^ data_m[f'msb{idx2}']
        data_m['lsb0'] = data_m['lsb0'] ^ data_m[f'lsb{idx2}']

    data_m['r'] = (
                (data_m['msb0'] * 1).astype(str) + (data_m['lsb0'] * 1).astype(
            str)).astype(int)
    data_m = data_m[['challenge', 'r']]
    data_m.to_csv(f'{folder}XOR/{n}bit_{sep_type}_ts{timestamp}_XOR0{"".join([str(i) for i in range(1, idx+1)])}.csv')
