import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from trading_signal import signal_gen_J, signal_gen_1
PATH1 = Path.cwd()


class Timing_backtest_minute():

    def __init__(self, signal, subject, omit_len = 20, interval = 5, initcap = 1e5):
        self.signal = signal
        self.initcap = initcap
        self.omit_len = omit_len
        self.interval = interval
        self.subject = self.get_subject(subject)

    def get_subject(self, subject):
        ret = pd.read_excel(PATH1 / subject, index_col=2, parse_dates=True)
        ret = ret.rename(columns={'开盘价(元)': 'open', '最高价(元)': 'high', '最低价(元)': 'low', '收盘价(元)': 'close'})
        ret.index.name = 'date'
        ret = ret[['open', 'high', 'low', 'close']]
        return ret

    def get_signal_net(self, start, end):

        close = self.subject['close']

        index = close.index
        data = self.subject[(index > pd.to_datetime(start)) & (index <= pd.to_datetime(end))]
        close = close[(index > pd.to_datetime(start)) & (index <= pd.to_datetime(end))]

        # signal = pd.Series(index=close.index, name='signal', dtype='float64')
        signal = pd.Series(name='signal', dtype='float64')
        net = pd.Series(index=close.index, name='net', dtype='float64')
        C = self.initcap


        cur = pd.to_datetime(start)
        ended = pd.to_datetime(end)
        delta = pd.Timedelta(days=1)
        ret = []
        while cur < ended:
            cur_date = cur.strftime('%Y-%m-%d')
            if cur_date in close.index:
                cur_close = close.loc[cur_date]
                cur_data = data.loc[cur_date]

                if not cur_data.empty:
                    signal = pd.concat([signal, self.signal(cur_data, self.omit_len, self.interval)])
                    prev_close = np.nan
                    position = 0.0
                    for i in range(len(cur_close)):
                        cur_signal = signal[i]
                        cur_index = cur_close.index[i]

                        if cur_signal == 1:
                            if not np.isnan(prev_close):
                                C = C + position * (cur_close.loc[cur_index] - prev_close)
                            position = C / cur_close.loc[cur_index]
                            prev_close = cur_close.loc[cur_index]
                        if cur_signal == -1:
                            if not np.isnan(prev_close):
                                C = C + position * (cur_close.loc[cur_index] - prev_close)
                            position = - C / cur_close.loc[cur_index]
                            prev_close = cur_close.loc[cur_index]
                        if cur_signal == 0:
                            if not np.isnan(prev_close):
                                C = C + position * (cur_close.loc[cur_index] - prev_close)
                            position = 0
                            prev_close = np.nan
                        # print(position)
                        net.loc[cur_index] = C
                    ret.append(self.signal(cur_data))

            cur += delta

        return close, signal, net

class Timing_backtest_day():
    def __init__(self, signal, subject, omit_len = 20, interval = 5, initcap = 1e5):
        self.signal = signal
        self.initcap = initcap
        self.omit_len = omit_len
        self.interval = interval
        self.subject = self.get_subject(subject)

    def get_subject(self, subject):
        ret = pd.read_excel(PATH1 / subject, index_col=2, parse_dates=True)
        ret = ret.rename(columns={'开盘价(元)': 'open', '最高价(元)': 'high', '最低价(元)': 'low', '收盘价(元)': 'close'})
        # ret = pd.read_excel(PATH1 / subject, index_col=0, parse_dates=True)
        # ret.index.name = 'date'
        ret = ret[['open', 'high', 'low', 'close']]
        return ret

    def get_signal_net(self, start, end):

        close = self.subject['close']
        index = close.index
        data = self.subject[(index > pd.to_datetime(start)) & (index <= pd.to_datetime(end))]
        close = close[(index > pd.to_datetime(start)) & (index <= pd.to_datetime(end))]

        signal = self.signal(data, self.omit_len, self.interval)
        net = pd.Series(index=close.index, name='net', dtype='float64')
        pnl = pd.Series(index=close.index, name='net', dtype='float64')
        C = self.initcap
        pnl[0] = C

        position = 0

        for i in range(len(close)):
            if i > 0:
                net[i] = position * (close[i] - close[i-1])
                C += net[i]
                pnl[i] = C

            if signal[i] == 1:
                position = C / close[i]

            elif signal[i] == -1:
                position = - C / close[i]

            elif signal[i] == 0:
                position = 0
                
        net.fillna(0, inplace=True)
        return close, signal, net, pnl

    def get_performance(self, signal, net, pnl):
        performance  = pd.Series(index = ['平均收益率', '标准差', '夏普比率', '最大回撤', '胜率', '盈亏比', '开仓次数'], dtype='float64')

        ret = pnl.pct_change().dropna()
        performance['平均收益率'] = ret.mean() * 252
        performance['标准差'] = ret.std() * np.sqrt(252)
        performance['夏普比率'] = performance['平均收益率'] / performance['标准差']
        c_max = pnl.iloc[0]
        mdd = 0
        for i in range(1, len(pnl)):
            c_max = max(c_max, pnl.iloc[i])
            mdd = max(mdd, c_max - pnl.iloc[i])
        performance['最大回撤'] = mdd / self.initcap

        prev_position = 0
        cur_pnl = 0
        signals = []
        per_pnl = []
        start_amount = []
        for i in range(len(signal)):
            if not np.isnan(signal[i]) and prev_position != signal[i]:
                if prev_position != 0:
                    signals.append(prev_position)
                    if signal[i] != 0:
                        start_amount.append(pnl[i])
                    prev_position = signal[i]
                    per_pnl.append(cur_pnl)
                    cur_pnl = 0
                else:
                    prev_position = signal[i]
                    start_amount.append(pnl[i])

            cur_pnl += net[i]

        if prev_position != 0:
            signals.append(prev_position)
            per_pnl.append(cur_pnl)

        performance['开仓次数'] = len(signals)
        win_num = 0
        loss_num = 0
        win_amount = 0
        loss_amount = 0
        for i in range(len(signals)):
            if per_pnl[i] > 0:
                win_num += 1
                win_amount += per_pnl[i] / start_amount[i]
            else:
                loss_num += 1
                loss_amount += per_pnl[i] / start_amount[i]
        if (win_num == 0) or (loss_num == 0):
            performance['盈亏比'] = np.nan
        else:
            performance['盈亏比'] = (win_amount/win_num) / abs((loss_amount/ loss_num))
        performance['胜率'] = win_num / performance['开仓次数']

        return signals, per_pnl, performance

    def start_backtest(self, start, end, eval_performance=True):
        _, signal, net, pnl = self.get_signal_net(start, end)
        signals, per_pnl, performance = None, None, None
        if eval_performance:
            signals, per_pnl, performance = self.get_performance(signal, net, pnl)
        return signals, per_pnl, performance


if __name__ == '__main__':
    # # 分钟频率
    # subject = '沪深300期货主力1分钟.xlsx'
    # subject = '中证500期货主力1分钟.xlsx'
    # df = pd.read_excel(PATH1 / subject, index_col=2, parse_dates=True)
    #
    # start = '2023-03-15'
    # end = '2023-09-11'
    #
    # backtest = Timing_backtest_minute(signal_gen_J, subject, omit_len=20, interval=3)
    # close, signal, net = backtest.get_signal_net(start, end)
    #
    # fig = plt.figure()
    # # net.plot()
    # plt.plot(net.values)
    # plt.title('profit and loss')
    # plt.show()
    #
    # fig1 = plt.figure()
    # plt.plot(close.values)
    # # close.plot()
    # plt.show()

    # # 日频率
    subject = '沪深300期货主力日数据.xlsx'
    # subject = 'test_df.xlsx'
    # df = pd.read_excel(PATH1 / subject, index_col=0, parse_dates=True)

    start = '2010-04-16'
    end = '2018-09-11'
    backtest = Timing_backtest_day(signal_gen_1, subject, omit_len=20, interval=1)

    close, signal, net, pnl = backtest.get_signal_net(start, end)
    signals, per_pnl, performance = backtest.get_performance(signal, net, pnl)

    fig = plt.figure()
    # net.plot()
    # plt.plot(net.values)
    plt.plot(pnl.values)
    plt.title('profit and loss')
    plt.show()

    fig = plt.figure()
    # net.plot()
    # plt.plot(net.values)
    plt.plot(net.values)
    plt.title('net')
    plt.show()

    fig1 = plt.figure()
    plt.plot(close.values)
    # close.plot()
    plt.show()

