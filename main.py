import argparse
import json
import numpy as np
import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from SwitchableStarModule import SwitchablestarModule
from modules.DataModule import PUFDataModule


def run_model(args, data_file, cbits, ids, name=""):
    train_ids, val_ids, test_ids = ids
    epochs = args.epochs
    hparams = args.hparams

    data_module = PUFDataModule(
        hparams['bs'],
        data_file,
        args.ts,
        cbits,
        train_ids,
        val_ids,
        test_ids
    )
    data_module.setup()

    logger = TensorBoardLogger('runs', name=f'logger{name}')

    monitor_params = {
        'monitor': 'Val Accuracy',
        'mode': "max"
    }
    early_stop_callback = EarlyStopping(
        min_delta=0.01,
        patience=40,
        verbose=False,
        stopping_threshold=0.95,
        **monitor_params
    )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        **monitor_params
    )
    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=epochs,
        logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_checkpointing=True
    )

    model = SwitchablestarModule(hparams, cbits, args.do_log)

    trainer.fit(model, datamodule=data_module)
    best_model = model.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.test(best_model, datamodule=data_module)

    train_accs = model.train_accs
    val_accs = model.val_accs
    test_accs = best_model.test_accs
    accs = (train_accs, val_accs, test_accs)

    if args.store_results:
        store_results(model, *accs, data_file, len(train_ids), args.ts)


def get_architecture_length(file):
    if '100k' in file:
        length = 100000
    elif '500k' in file:
        length = 499733
    elif '12bit' in file:
        length = 4096
    elif '_sep_' in file or '_rand_inst' in file:
        length = 500000
    else:
        raise RuntimeError(f'Size of dataset {file} could not be inferred.')
    return length


def run_on_all_data(args, root, files):
    for file in files:
        file_names = file.split('/')
        # Only run if there is so corresponding saved model
        if not os.path.isfile(get_store_name_of_model(file_names)):
            length = get_architecture_length(file)
            ids = list(range(length))
            np.random.shuffle(ids)
            train_ids = ids[:int(length * 0.8)]
            val_ids = ids[len(train_ids):int(length * 0.9)]
            test_ids = ids[len(train_ids) + len(val_ids):]
            data_dir = f'{root}/{file}'

            cbits = int(file.split('bit')[0])
            with open('hparams.json', 'r') as hparam_f:
                all_hparams = json.load(hparam_f)
                hparams = all_hparams[args.a][str(cbits)]
            args.hparams = hparams

            run_model(args, data_dir, cbits,
                      (train_ids, val_ids, test_ids), name=file)
        else:
            print("A model for run on", file, "already exists.")


def run_different_sizes(args, cbits):
    data_dir = f'data/{args.a}/{args.f}.csv'
    file_names = data_dir.split('/')

    length = get_architecture_length(args.f)
    ids = list(range(length))
    np.random.shuffle(ids)

    start = length // 5
    train_end = int(0.8 * length)
    val_test_size = int(0.1 * length)
    train_sizes = list(range(start, train_end+1, start))

    for size in train_sizes:
        # Only run if there is so corresponding saved data
        with open(f'storage/results.json', 'r') as file:
            results = json.load(file)
            arch_name = file_names[1]
            file_name = file_names[2]

            if not (arch_name in results and
                    file_name in results[arch_name] and
                    str(args.ts) in results[arch_name][file_name] and
                    str(size) in results[arch_name][file_name][str(args.ts)]):
                train_ids = ids[:size]
                val_ids = ids[train_end:train_end + val_test_size]
                test_ids = ids[train_end + val_test_size:]
                run_model(args, data_dir, cbits,
                          (train_ids, val_ids, test_ids),
                          name=f'{args.f}_tl{size}')
            else:
                print("A model for run on", f'{args.f}_tl{size}', "already exists.")


def run_on_all_pufs(args):
    # root, _, files = next(os.walk('data/ArbiterStarPUF2'))
    # run_on_all_data(args, root, files)

    # root2, _, files2 = next(os.walk('data/SwitchableStarPUF'))
    # run_on_all_data(args, root2, files2)

    root3, _, files3 = next(os.walk('data/SwitchableStarPUFXOR'))
    run_on_all_data(args, root3, files3)


def get_store_name_of_model(file_names):
    return f'storage/models/{file_names[1]}_{file_names[2]}_model.pt'


def store_results(model, train_accs, val_accs, test_accs, data_file,
                  train_size, timestamp):
    file_names = data_file.split('/')
    torch.save(
        model.state_dict(), get_store_name_of_model(file_names)
    )
    timestamp = str(timestamp)
    train_size = str(train_size)
    with open(f'storage/results.json', 'r+') as file:
        results = json.load(file)
        arch_name = file_names[1]
        file_name = file_names[2]
        if arch_name not in results:
            results[arch_name] = {}
        if file_name not in results[arch_name]:
            results[arch_name][file_name] = {}
        if timestamp not in results[arch_name][file_name]:
            results[arch_name][file_name][timestamp] = {}
        results[arch_name][file_name][timestamp][train_size] = {
            'acc': {
                'train': np.max(train_accs),
                'val': np.max(val_accs),
                'test': test_accs
            }
        }
        file.seek(0)
        json.dump(results, file)
        file.truncate()


def main(args):
    cbits = int(args.f.split('bit')[0])

    with open('hparams.json', 'r') as hparam_f:
        all_hparams = json.load(hparam_f)
        hparams = all_hparams[args.a][str(cbits)]
    args.hparams = hparams

    np.random.seed(345)

    if args.ab:
        # Add run for additional bit
        file_latter_part = args.f.split('bit')[1]
        f2 = args.ab + 'bit' + file_latter_part

        run_different_sizes(args, cbits)
        args.f = f2
        run_different_sizes(args, cbits)
        exit()

    run_different_sizes(args, cbits)
    # run_on_all_pufs(args)
    exit()

    data_dir = f'data/{args.a}/{args.f}.csv'
    length = get_architecture_length(args.f)
    ids = list(range(length))
    np.random.shuffle(ids)
    train_ids = ids[:int(length * 0.9)]
    val_ids = ids[len(train_ids):int(length * 0.95)]
    test_ids = ids[len(train_ids) + len(val_ids):]
    run_model(args, data_dir, cbits, (train_ids, val_ids, test_ids),
              name=args.f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', '--architecture', default='SwitchableStarPUF_adjC')
    parser.add_argument('--f', '--file', default='64bit_rand_inst0_0_sep')
    # parser.add_argument('--f', '--file', default='12bit_enumerate_0')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--store-results', default=True)
    parser.add_argument('--do-log', default=True)
    parser.add_argument('--ts', '--timestamp', type=str, default=10)
    parser.add_argument('--ab', '--add_bit', type=str, default=None)
    # parser.add_argument('--timestamp', default=75)

    args = parser.parse_args()
    main(args)
