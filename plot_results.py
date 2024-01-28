import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


def run_data_class_count():
    with open(f'storage/data_analysis.json', 'r+') as f:
        data_analysis = json.load(f)
        root1, _, files1 = next(os.walk('data/ArbiterStarPUF2'))
        root2, _, files2 = next(os.walk('data/SwitchableStarPUF'))

        for root, files in [(root1, files1), (root2, files2)]:
            for file in tqdm(files):
                data_dir = f'{root}/{file}'
                file_names = data_dir.split('/')
                data = pd.read_csv(data_dir)

                cnts = np.unique(data[[data.columns[203]]], return_counts=True)
                data_analysis[file_names[1]][file_names[2]] = {
                    '00': int(cnts[1][0]),
                    '01': int(cnts[1][1]),
                    '10': int(cnts[1][2]),
                    '11': int(cnts[1][3])
                }
        f.seek(0)
        json.dump(data_analysis, f)
        f.truncate()


def run_data_class_count_diff_timestamps_interactive():
    timestamps = list(range(75, 100, 1))
    timestamps = list(range(0, 51, 10))
    data_dir = f'data/SwitchableStarPUF/128bit_rand_inst0_avg_sep.csv'
    data = pd.read_csv(data_dir)
    for timestamp in tqdm(timestamps):
        _, counts = np.unique(data[[data.columns[timestamp]]],
                              return_counts=True)

        df = pd.DataFrame({
            '00': counts[0],
            '01': counts[1],
            '10': counts[2],
            '22': counts[3]
        }, index=[0])
        g = sns.barplot(df)
        plt.subplots_adjust(hspace=0.4, wspace=0.2)
        plt.subplots_adjust(hspace=0.4, wspace=0.2)
        plt.title(timestamp)
        plt.show()


def plot_acc_heatmap_diff_sizes():
    with open('storage_diff_size/results.json', 'r') as f:
        data = json.load(f)
    keys = [f'12bit_enumerate_{idx}.csv' for idx in range(10)] + [
        '32bit_rand100k.csv', '64bit_rand500k.csv'] + [
               f'64bit_rand100k_{idx}.csv'
               for idx in range(10)]
    keys = ['64bit_rand500k.csv']
    # sizes3 = list(range(10000, 100000, 10000))
    # sizes2 = list(range(10000, 20000, 1000))
    # sizes1 = list(range(1000, 10000, 1000))
    # sizes = sizes1 + sizes2 #+ sizes3

    switchable_vals = []
    sizes = []
    for key in keys:
        data = list(data['SwitchableStarPUF'][key].items())
        for size, result in data:
            value = result['acc']['test']
            switchable_vals.append(value)
            sizes.append(size)
    # arbiter_vals = [data['ArbiterStarPUF2'][key]['acc']['test'] for key in keys]
    # switchable_vals = [data['SwitchableStarPUF']['32bit_rand100k.csv'][str(size)]['acc']['test'] for size in sizes]

    fig, ax = plt.subplots(figsize=(15, 2))
    plt.margins(10)
    sns.heatmap(np.array(switchable_vals)[np.newaxis, :], vmin=0, vmax=1,
                annot=True, xticklabels=sizes, cmap="Blues")
    plt.tight_layout()
    plt.show()
    # fig.savefig('results_sizes.jpg')


def plot_acc_heatmap_diff_archs():
    with open('storage/results.json', 'r') as f:
        data = json.load(f)
    keys = [f'12bit_enumerate_{idx}.csv' for idx in range(10)] + \
           ['32bit_rand100k.csv'] + \
           [f'64bit_rand100k_{idx}.csv' for idx in range(10)] + \
           ['64bit_rand500k.csv']
    keys = [f'{bits}bit_{sep}_sep_ts10_XOR{idx}.csv' for bits in
            [32, 64, 96, 128] for idx in ['01', '012', '0123'] for sep in
            ['0', 'avg']]
    x_plt_keys = ['20%', '40%', '60%', '80%']
    y_plt_keys = [f'{bits}bit_{len(idx)}-XOR ({sep})' for bits in
                  [32, 64, 96, 128] for idx in ['01', '012', '0123'] for sep in
                  ['0', 'avg']]

    switchable_vals = []
    for key in keys:
        all_vals = []
        for train_size, values in list(
                data['SwitchableStarPUFXOR'][key].items()):
            # results = list(data['SwitchableStarPUFXOR'][key].items())[0][-1]
            value = values['acc']['test']
            all_vals.append(value)
        switchable_vals.append(np.array(all_vals))

    fig, ax = plt.subplots(figsize=(10, 2.5))
    plt.margins(10)
    '''sns.heatmap(np.array(switchable_vals)[np.newaxis, :], vmin=0, vmax=1,
                annot=True, xticklabels=plt_keys, cmap="Blues")'''
    sns.heatmap(np.array(switchable_vals), vmin=0, vmax=1,
                annot=True, xticklabels=x_plt_keys, yticklabels=y_plt_keys,
                cmap="Blues")
    plt.tight_layout()
    plt.show()
    fig.savefig('results_archs.jpg')


def plot_acc_lines_diff_archs():
    with open('storage/results.json', 'r') as f:
        data = json.load(f)

    data_folder = 'SwitchableStarPUF_newInput'
    #data_folder = 'SwitchableStarPUF'
    data_folderXOR = f"{data_folder}XOR"
    timestamp = '97'
    separators = ['0', 'avg']
    separators = ['0']
    nbits = [32, 64, 96]
    instances = [0]

    keys_inst = [f'{bits}bit_rand_inst{x}_{sep}_sep.csv' for bits in
            nbits for sep in separators for x in instances]
    """keys_xor = [f'{bits}bit_{sep}_sep_ts{timestamp}_XOR{idx}.csv' for
                bits in nbits for idx in ['01', '012', '0123'] for sep in
                separators]"""

    keys_xor = [f'{bits}bit_{sep}_sep_ts{timestamp}_XOR{idx}.csv' for
                bits in nbits for idx in ['01', '012', '0123'] for sep in
                separators]

    """keys_xor += [f'64bit_{sep}_sep_ts{timestamp}_XOR{idx}_{nbit}orig.csv' for
                nbit in [96, 128] for idx in ['01', '012', '0123'] for sep in
                separators]"""

    x_plt_keys = ['20%', '40%', '60%', '80%']
    y_plt_keys = [f'No XOR (inst{x}, {sep})' for x in instances for sep in
                  separators] + [
                     f'{len(idx)}-XOR ({sep})' for idx in ['01', '012', '0123']
                     for sep in separators]

    #y_plt_keys += [f'{len(idx)}-XOR ({sep})' for idx in ['01', '012', '0123']
    #               for sep in separators]

    """+ [f'(64bit) {len(idx)}-XOR ({sep})'
                                             for idx in ['01', '012', '0123']
                                             for sep in separators]"""

    switchable_vals = {}
    all_keys = [keys_inst, keys_xor]
    # all_keys = [ keys_xor]
    all_folders = [data_folder, data_folderXOR]
    # all_folders = [data_folderXOR]
    for keys, folder in zip(all_keys, all_folders):
        for key in keys:
            print(key)
            poss_keys = data[folder]
            bits = key.split('bit')[0]
            if bits not in switchable_vals:
                switchable_vals[bits] = []
            if not key in poss_keys:
                warnings.warn(f'Key {key} does not exist')
                switchable_vals[bits].append([0,0,0,0])
                continue
            if "ts" in key and timestamp not in list(data[folder][key].keys()) and key.split('ts')[1].split("_")[0] == timestamp:
                vals = data[folder][key].items()
            else:
                try:
                    vals = data[folder][key][timestamp].items()
                except Exception as e:
                    print(f'---------------- Timestamp {timestamp} does not exist!')
                    #if nbits[0] > 32:
                    #    continue
                    vals = data[folder][key].items()

            train_vals = []
            for train_size, values in vals:
                try:
                    value = values['acc']['test']
                    train_vals.append(value)
                except Exception as e:
                    print(e)
                    print("Probably timestamp doesn't exist")
                    train_vals.extend([0,0,0,0])
                    break
            #if 'orig' in key:
            #    bits = key.split('orig')[0].split('_')[-1]
            switchable_vals[bits].append(train_vals)


    for key, vals in switchable_vals.items():
        print(key, timestamp)
        for val, name in zip(vals, y_plt_keys):
            print(name, val[-1])

    #exit()


    num_subplots = len(switchable_vals)
    num_cols = 2
    num_rows = 2

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    axs = axs.flatten()
    plt.suptitle(f"Timestamp Idx: {timestamp} - {data_folder}")

    for i, bits in enumerate(list(switchable_vals.keys())[:num_subplots]):
        for vals in switchable_vals[bits]:
            axs[i % 4].plot(x_plt_keys, np.array(vals))
        axs[i % 4].set_xlabel('Train Size')
        axs[i % 4].set_ylabel('Accuracy')
        axs[i % 4].set_title(f'{bits}bit')
        axs[i % 4].set_ylim(0.25, 1)
        for tick in axs[i % 4].get_yticks():
            axs[i % 4].axhline(tick, linestyle='dotted', color='gray')

    # Add legend
    for i, bits in enumerate(list(switchable_vals.keys())[:num_subplots]):
        labels = [key for key in y_plt_keys]
        if i > 0:
            continue
        axs[i % 4].legend(labels, loc='lower right')

    plt.tight_layout()
    fig.savefig('new.jpg')
    plt.show()

plot_acc_lines_diff_archs()
'''def plot_output_distr(interactive=False):
    with open('storage/results.json', 'r') as f:
        data = json.load(f)
    keys = [f'12bit_enumerate_{idx}.csv' for idx in range(10)] + [
        '32bit_rand100k.csv', '64bit_rand500k.csv'] + [
               f'64bit_rand100k_{idx}.csv'
               for idx in range(10)]
    archs = [
        'SwitchableStarPUF',
        # 'ArbiterStarPUF2'
    ]
    classes = ['00', '01', '10', '11']
    with open(f'storage/data_analysis.json', 'r+') as f:
        data = json.load(f)

    for arch in archs:
        counts = [[key] + [data[arch][key][cls] for cls in classes] for key in
                  keys]

        df = pd.DataFrame(counts, columns=['name'] + classes)
        g = sns.catplot(df, col='name', kind='bar', sharey=False, col_wrap=4,
                        height=2, aspect=16 / 9)
        plt.subplots_adjust(hspace=0.4, wspace=0.2)
        if interactive:
            plt.show()
        else:
            g.savefig(f'{arch}_output_distr.jpg')


def plot_acc_heatmap_XORs():
    with open('storage/results.json', 'r') as f:
        data = json.load(f)
    keys = ["12bit_XOR" + "".join(str(j) for j in range(i + 1)) + ".csv" for i
            in
            range(1, 10)]
    plt_keys = [f'12 bit {idx}-XOR' for idx in range(2, 11)]

    switchable_vals = []
    for key in keys:
        results = list(data['SwitchableStarPUFXOR'][key].items())[-1][1]
        value = results['acc']['test']
        switchable_vals.append(value)

    fig, ax = plt.subplots(figsize=(15, 2))
    plt.margins(10)
    sns.heatmap(np.array(switchable_vals)[np.newaxis, :], vmin=0, vmax=1,
                annot=True, xticklabels=plt_keys, cmap="Blues")
    plt.tight_layout()
    plt.show()
    fig.savefig('results_sizes_XOR.jpg')

    ###
    keys = [f'12bit_enumerate_{idx}.csv' for idx in range(10)]
    plt_keys = [f'12 bit [{idx}]' for idx in range(10)]

    switchable_vals = []
    for key in keys:
        results = list(data['SwitchableStarPUF'][key].items())[-1][1]
        value = results['acc']['test']
        switchable_vals.append(value)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    plt.margins(10)
    sns.heatmap(np.array(switchable_vals)[np.newaxis, :], vmin=0, vmax=1,
                annot=True, xticklabels=plt_keys, cmap="Blues")
    plt.tight_layout()
    plt.show()

    ###
    keys = [f'32bit_sr500k_{idx}.csv' for idx in range(4)]
    plt_keys = [f'32 bit [{idx}]' for idx in range(4)]

    switchable_vals = []
    for key in keys:
        results = list(data['SwitchableStarPUF'][key].items())[-1][1]
        value = results['acc']['test']
        switchable_vals.append(value)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    plt.margins(10)
    sns.heatmap(np.array(switchable_vals)[np.newaxis, :], vmin=0, vmax=1,
                annot=True, xticklabels=plt_keys, cmap="Blues")
    plt.tight_layout()
    plt.show()

    keys = ["32bit_0XOR1sr500k_0.csv", "32bit_0XOR1XOR2sr500k_0.csv",
            "32bit_0XOR1XOR2XOR3sr500k_0.csv"]
    plt_keys = ['2-XOR', '3-XOR', '4-XOR']

    switchable_vals = []
    for key in keys:
        results = list(data['SwitchableStarPUFXOR'][key].items())[-1][1]
        value = results['acc']['test']
        switchable_vals.append(value)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    plt.margins(10)
    sns.heatmap(np.array(switchable_vals)[np.newaxis, :], vmin=0, vmax=1,
                annot=True, xticklabels=plt_keys, cmap="Blues")
    plt.tight_layout()
    plt.show()


# plot_acc_heatmap_XORs()
# exit()
# run_data_class_count_diff_timestamps_interactive()
plot_acc_lines_diff_archs()
# plot_acc_heatmap_diff_archs()
# run_data_class_count_diff_timestamps_interactive()
'''