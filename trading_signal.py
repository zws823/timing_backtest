from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from filter import fourier_filter
import pandas as pd
import talib as ta
from statsmodels.tsa.filters.hp_filter import hpfilter

PATH1 = Path.cwd()

def signal_gen_J(data, omit_len = 20, interval = 5):
    K, D = ta.STOCH(data['high'], data['low'], data['close'], fastk_period=30, slowk_period=3, slowd_period=3)
    K = K / 100
    D = D / 100
    J = 3 * K - 2 * D

    fig, ax1 = plt.subplots(figsize=(30, 10))
    ax2 = ax1.twinx()
    ax1.plot(K.values, '-r', label='K')
    ax1.plot(D.values, '-b', label='D')
    # ax1.plot(J.values, '-c', label='J')
    ax2.plot(data['close'].values, '-g', label='close price')
    plt.legend()
    plt.show()

    signal = pd.Series(index=K.index, name='signal', dtype='float64')
    length = len(signal)

    prev = 0
    tau = 0.2
    # 0是平仓，1是开多仓，-1是开空仓
    for i in range(omit_len, length, interval):
        # print(J[i])
        if prev == 0:
            if J[i] > 1:
                signal[i] = 1
                prev = -1
            elif J[i] < 0:
                signal[i] = -1
                prev = 1
            else:
                signal[i] = 0
                prev = 0

        elif prev == -1:
            if J[i] > 1 - tau:
                signal[i] = 1
                prev = -1
            elif J[i] < 0:
                signal[i] = -1
                prev = 1
            else:
                signal[i] = 0
                prev = 0

        elif prev == 1:
            if J[i] > 1:
                signal[i] = 1
                prev = -1
            elif J[i] < tau:
                signal[i] = -1
                prev = 1
            else:
                signal[i] = 0
                prev = 0
    # signal[-1] = 0
    return signal

def signal_gen_1(data, *args):
    H_line, M_line, L_line = ta.BBANDS(data['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    close = data['close']

    signal = pd.Series(index = data.index, name = 'signal', dtype = 'float64')
    length = len(signal)
    flag = 0

    for i in range(length):
        if i < 20:
            continue
        if flag != -1 and close[i-1] > H_line[i-1]:
            signal[i] = -1
            flag = -1
        if flag != 1 and close[i-1] < L_line[i-1]:
            signal[i] = 1
            flag = 1
        if flag == 1 and close[i-1] > M_line[i-1]:
            signal[i] = 0
            flag = 0
        if flag == -1 and close[i-1] < M_line[i-1]:
            signal[i] = 0
            flag = 0

    return signal

def signal_gen_2(data, *args):
    H_line, M_line, L_line = ta.BBANDS(data['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    close = data['close']
    low = data['low']
    high = data['high']
    threshold = 0.05

    signal = pd.Series(index = data.index, name = 'signal', dtype = 'float64')
    length = len(signal)
    flag = 0
    openflag = True

    for i in range(length):
        if i < 20:
            continue
        if flag != -1 and close[i-1] > H_line[i-1] and openflag:
            signal[i] = -1
            reset_price = close[i-1] * (1 + threshold)
            flag = -1
        if flag != 1 and close[i-1] < L_line[i-1] and openflag:
            signal[i] = 1
            reset_price = close[i-1] * (1 - threshold)
            flag = 1
        if flag == 1 and close[i-1] > M_line[i-1]:
            if openflag:
                signal[i] = 0
                flag = 0
            else:
                flag = 0
            openflag = True
        if flag == 1 and low[i-1] < reset_price and openflag:
            signal[i] = 0
            openflag = False
        if flag == -1 and close[i-1] < M_line[i-1]:
            if openflag:
                signal[i] = 0
                flag = 0
            else:
                flag = 0
            openflag = True
        if flag == -1 and high[i-1] > reset_price and openflag:
            signal[i] = 0
            openflag = False

    return signal

def signal_gen_2(data, *args):
    H_line, M_line, L_line = ta.BBANDS(data['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    close = data['close']
    vol = close.pct_change().rolling(20).std()
    vol_meidan = vol.rolling(20).median()

    signal = pd.Series(index = data.index, name = 'signal', dtype = 'float64')
    length = len(signal)
    flag = 0

    for i in range(length):
        if i < 20:
            continue
        if flag != -1 and close[i-1] > H_line[i-1] and vol[i-1] > vol_meidan[i-1]:
            signal[i] = -1
            flag = -1
        if flag != 1 and close[i-1] < L_line[i-1] and vol[i-1] > vol_meidan[i-1]:
            signal[i] = 1
            flag = 1
        if flag == 1 and close[i-1] > M_line[i-1]:
            signal[i] = 0
            flag = 0
        if flag == -1 and close[i-1] < M_line[i-1]:
            signal[i] = 0
            flag = 0

    return signal

def signal_gen_3(data, *args):
    _, _, macd = ta.MACD(data['close'])

    signal = pd.Series(index = data.index, name='signal', dtype='float64')
    length = len(signal)
    flag = 0
    tau = 20

    for i in range(length):
        if i < 15:
            continue
        if flag == 0:
            if macd[i-1] > tau:
                signal[i] = -1
                flag = -1
            if macd[i-1] < -tau:
                signal[i] = 1
                flag = 1
        if flag == -1:
            if macd[i-1] < -tau:
                signal[i-1] = 1
                flag = 1
            if macd[i-1] >= -tau and macd[i-1] < 0:
                signal[i-1] = 0
                flag = 0
        if flag == 1:
            if macd[i-1] > tau:
                signal[i-1] = -1
                flag = -1
            if macd[i-1] <= tau and macd[i-1] > 0:
                signal[i-1] = 0
                flag = 0

    return signal

if __name__ == '__main__':
    subject = '沪深300期货主力日数据.xlsx'
    df = pd.read_excel(PATH1 / subject, index_col=2, parse_dates=True)
    df = df.rename(columns={'开盘价(元)':'open', '最高价(元)':'high', '最低价(元)':'low', '收盘价(元)':'close'})
    df.index.name = 'date'
    df = df[['open', 'high', 'low', 'close']]

    # date = '2023-03-14'
    # df1 = df.loc[date]
    # signal = signal_gen_J(df1)

    start = '2010-04-16'
    end = '2010-11-11'
    signal = signal_gen_1(df.loc[start:end])

    # # 创建或导入信号
    # # 这里我们创建一个含有两个不同频率的简单信号
    # fs = 1000  # 采样频率
    # t = np.arange(0, 1, 1 / fs)  # 时间轴
    # freq1, freq2 = 5, 50  # 频率值
    # signal = np.sin(2 * np.pi * freq1 * t) + np.sin(2 * np.pi * freq2 * t)
    #
    # filtered_signal = fourier_filter(signal, 0.005)
    #
    # # 可视化结果
    # plt.figure(figsize=(12, 6))
    # plt.subplot(2, 1, 1)
    # plt.plot(t, signal)
    # plt.title('Original Signal')
    # plt.subplot(2, 1, 2)
    # plt.plot(t, filtered_signal)
    # plt.title('Filtered Signal')
    # plt.tight_layout()
    # plt.show()





