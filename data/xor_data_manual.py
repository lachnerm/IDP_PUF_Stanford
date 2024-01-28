import pandas as pd

data_file0 = 'SwitchableStarPUF/32bit_sr500k_0.csv'
data_file1 = 'SwitchableStarPUF/32bit_sr500k_1.csv'
data_file2 = 'SwitchableStarPUF/32bit_sr500k_2.csv'
data_file3 = 'SwitchableStarPUF/32bit_sr500k_3.csv'
timestamp = 75
cbits = 32

data0 = pd.read_csv(data_file0)
data0.columns = ['challenge'] + list(range(1, len(data0.columns)))
data0 = data0[['challenge', timestamp]]
data0.columns = ['challenge', 0]
data0['msb0'] = data0[0] >= 10
data0['lsb0'] = data0[0] % 2 == 1

data1 = pd.read_csv(data_file1)
data1.columns = ['challenge'] + list(range(1, len(data1.columns)))
data1 = data1[['challenge', timestamp]]
data1.columns = ['challenge', 1]
data1['msb1'] = data1[1] >= 10
data1['lsb1'] = data1[1] % 2 == 1

data2 = pd.read_csv(data_file2)
data2.columns = ['challenge'] + list(range(1, len(data2.columns)))
data2 = data2[['challenge', timestamp]]
data2.columns = ['challenge', 2]
data2['msb2'] = data2[2] >= 10
data2['lsb2'] = data2[2] % 2 == 1

data3 = pd.read_csv(data_file3)
data3.columns = ['challenge'] + list(range(1, len(data3.columns)))
data3 = data3[['challenge', timestamp]]
data3.columns = ['challenge', 3]
data3['msb3'] = data3[3] >= 10
data3['lsb3'] = data3[3] % 2 == 1

data_m = data0.merge(data1, on='challenge').merge(data2, on='challenge').merge(data3, on='challenge')
data_m['msb'] = data_m['msb0'] ^ data_m['msb1'] ^ data_m['msb2'] ^ data_m['msb3']
data_m['lsb'] = data_m['lsb0'] ^ data_m['lsb1'] ^ data_m['lsb2'] ^ data_m['lsb3']

data_m['r'] = ((data_m['msb'] * 1).astype(str) + (data_m['lsb'] * 1).astype(
    str)).astype(int)
data_m = data_m[['challenge', 'r']]
data_m.to_csv('SwitchableStarPUFXOR/32bit_0XOR1XOR2XOR3sr500k_0.csv')
