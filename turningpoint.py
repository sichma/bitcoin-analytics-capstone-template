# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 13:31:20 2026

@author: GASH000
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
from scipy import stats
import os
from datetime import datetime, timedelta

# 设置tushare token（请替换为你的token）
ts.set_token('0eaedd58fde7cd9bb30751d00e3216de97b287b6e1adb55d1fd06edc')
pro = ts.pro_api()

def rolling_linear_regression(series, window):
    """
    对每个点计算基于该点及之前window-1个点的线性回归斜率和R²。
    返回两个Series，索引与series相同。
    """
    slopes = pd.Series(index=series.index, dtype=float)
    rsqs = pd.Series(index=series.index, dtype=float)
    for i in range(len(series)):
        if i < window - 1:
            continue
        y = series.iloc[i-window+1:i+1].values
        x = np.arange(len(y))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        slopes.iloc[i] = slope
        rsqs.iloc[i] = r_value**2
    return slopes, rsqs

def find_local_extrema(close):
    """
    寻找局部极值点：当前点小于等于前后各两个点中的最小值，或大于等于最大值。
    返回DataFrame，包含日期和价格。
    """
    tp = []
    for i in range(2, len(close)-2):
        left = close.iloc[i-2:i+1]  # i-2, i-1, i
        right = close.iloc[i+1:i+3] # i+1, i+2
        if close.iloc[i] <= min(left.min(), right.min()) or close.iloc[i] >= max(left.max(), right.max()):
            tp.append({'date': close.index[i], 'close': close.iloc[i]})
    return pd.DataFrame(tp)

def adjust_turning_points(tpts, noise_width=2.0):
    """
    调整转折点：合并相邻同向段和噪声段。
    tpts: DataFrame with columns 'date', 'close', sorted.
    noise_width: 忽略的收益率阈值（%）
    返回调整后的tpts。
    """
    # 确保包含首尾
    if len(tpts) == 0:
        return tpts
    first_date = tpts['date'].iloc[0]
    last_date = tpts['date'].iloc[-1]
    
    # 计算每段收益率
    def calc_returns(t):
        if len(t) < 2:
            return pd.Series(dtype=float)
        ret = t['close'].pct_change().dropna() * 100
        ret.index = t['date'].iloc[1:]  # 收益率对应每段的结束日期
        return ret
    
    tpts = tpts.copy().sort_values('date').reset_index(drop=True)
    while True:
        if len(tpts) < 2:
            break
        tpr = calc_returns(tpts)
        if len(tpr) < 1:
            break
        # 检查是否有相邻同向或小收益
        prod = tpr.iloc[:-1].values * tpr.iloc[1:].values  # 相邻乘积
        has_same_sign = np.any(prod > 0)
        has_small = np.any(np.abs(tpr) < noise_width)
        if not (has_same_sign or has_small):
            break
        
        # 构建段信息
        segments = []
        for i in range(len(tpr)):
            sd = tpts.iloc[i]['date']
            ed = tpts.iloc[i+1]['date']
            r = tpr.iloc[i]
            s = 'P' if r >= 0 else 'N'
            segments.append({'sd': sd, 'ed': ed, 'r': r, 's': s})
        
        # 合并相邻同向段：删除中间转折点
        to_delete = set()
        for sign in ['P', 'N']:
            same = [seg for seg in segments if seg['s'] == sign]
            for j in range(1, len(same)):
                if same[j-1]['ed'] == same[j]['sd']:
                    to_delete.add(same[j-1]['ed'])
        
        if to_delete:
            tpts = tpts[~tpts['date'].isin(to_delete)].reset_index(drop=True)
            continue  # 重新计算tpr
        
        # 合并小收益段：删除结束点
        to_delete_small = set()
        for i, r in tpr.items():
            if abs(r) < noise_width:
                to_delete_small.add(i)  # i是结束日期
        if to_delete_small:
            tpts = tpts[~tpts['date'].isin(to_delete_small)].reset_index(drop=True)
            continue
        
        break  # 没有变化则退出
    
    # 确保首尾存在
    if tpts['date'].iloc[0] != first_date:
        tpts = pd.concat([pd.DataFrame({'date': [first_date], 'close': [tpts['close'].iloc[0]]}), tpts], ignore_index=True)
    if tpts['date'].iloc[-1] != last_date:
        tpts = pd.concat([tpts, pd.DataFrame({'date': [last_date], 'close': [tpts['close'].iloc[-1]]})], ignore_index=True)
    tpts = tpts.drop_duplicates('date').sort_values('date').reset_index(drop=True)
    return tpts

def get_tps_lt_exp(ts, r_threshold=0.6, regression_window=5, noise_width=2.0):
    """
    主函数：识别转折点并计算分段统计。
    ts: DataFrame with DatetimeIndex and 'close' column.
    返回 (tpts, result)
    tpts: DataFrame with columns 'date', 'close'
    result: DataFrame with columns ['sdate', 'edate', 'tdays', 'tgap', 'tr', 'tmdd']
    """
    close = ts['close']
    
    # 第一步：局部极值候选点
    candidates = find_local_extrema(close)
    
    # 第二步：滚动线性回归
    slopes, rsqs = rolling_linear_regression(close, regression_window)
    
    # 筛选满足条件的候选点
    tpts_list = []
    for _, row in candidates.iterrows():
        tpd = row['date']
        # 找到该日期在slopes中的位置
        pos = slopes.index.get_loc(tpd)
        if pos is None:
            continue
        if pos + 4 >= len(slopes):
            continue
        ls = slopes.iloc[pos]
        lr2 = rsqs.iloc[pos]
        rs = slopes.iloc[pos + 4]
        rr2 = rsqs.iloc[pos + 4]
        if pd.isna(ls) or pd.isna(rs):
            continue
        if max(lr2, rr2) >= r_threshold and ls * rs <= 0:
            tpts_list.append({'date': tpd, 'close': close.loc[tpd]})
    
    # 加入首尾
    tpts_list.append({'date': close.index[0], 'close': close.iloc[0]})
    tpts_list.append({'date': close.index[-1], 'close': close.iloc[-1]})
    tpts = pd.DataFrame(tpts_list).drop_duplicates('date').sort_values('date').reset_index(drop=True)
    
    # 调整转折点
    tpts = adjust_turning_points(tpts, noise_width)
    
    # 计算每个段的信息
    result = []
    for i in range(len(tpts)-1):
        tsd = tpts.iloc[i]['date']
        ted = tpts.iloc[i+1]['date']
        sub = close.loc[tsd:ted]
        if len(sub) < 2:
            continue
        
        # 子段日收益率
        subr = sub.pct_change() * 100
        subr.iloc[0] = 0  # 第一天的收益率为0
        
        # 总收益率
        tr = (sub.iloc[-1] / sub.iloc[0] - 1) * 100
        
        # 翻转收益率（如果总收益为负）
        if tr < 0:
            subr = -subr
        
        # 累积路径
        subc = 100 * (1 + subr/100).cumprod()
        # 最大回撤（取正数）
        dd = (subc / subc.cummax() - 1).min()
        tmdd = -dd * 100 if dd < 0 else 0  # 转换为正百分比
        
        # 线性路径（等间距）
        x_vals = np.linspace(0, 1, len(sub))
        linear_prices = sub.iloc[0] + x_vals * (sub.iloc[-1] - sub.iloc[0])
        linear_returns = pd.Series(linear_prices).pct_change() * 100
        linear_returns.iloc[0] = 0
        
        # print(subr, linear_returns)
        # 计算tgap（均方根误差）
        tgap = np.sqrt(np.mean((subr - list(linear_returns))**2))
        
        result.append({
            'sdate': tsd,
            'edate': ted,
            'tdays': len(sub),
            'tgap': tgap,
            'tr': tr,
            'tmdd': tmdd
        })
    
    result_df = pd.DataFrame(result)
    return tpts, result_df

# 测试脚本
def run_test(mkt_name, data_symbol, start_date, split_date, end_date, output_dir):
    global res1
    """
    获取数据，对两个时间段分别运行分析，并生成图表。
    """
    # 获取数据（tushare示例，根据实际情况调整）
    if mkt_name == 'CSI300':
        df = pro.index_daily(ts_code='000300.SH', start_date=start_date, end_date=end_date)
    elif mkt_name == 'ZZ500':
        df = pro.index_daily(ts_code='000905.SH', start_date=start_date, end_date=end_date)
    elif mkt_name == 'SP500':
        df = pro.index_daily(ts_code='.SPX', start_date=start_date, end_date=end_date)
    elif mkt_name == 'Russell2000':
        df = pro.index_daily(ts_code='^RUT', start_date=start_date, end_date=end_date)
    elif mkt_name == 'NHCI':
        df = pro.index_daily(ts_code='^NHCI', start_date=start_date, end_date=end_date)
    else:
        raise ValueError("Unsupported market")
    
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.set_index('trade_date').sort_index()
    df = df[['close']]
    
    # 分割数据
    split = pd.to_datetime(split_date)
    df1 = df.loc[:split]
    df2 = df.loc[split:]
    
    # 运行函数
    tpts1, res1 = get_tps_lt_exp(df1)
    tpts2, res2 = get_tps_lt_exp(df2)
    
    # 生成图表
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # 第一行：前半段
    ax = axes[0, 0]
    ax.plot(df1.index, df1['close'], label='Close')
    ax.scatter(tpts1['date'], tpts1['close'], color='red', s=30, label='Turning Points')
    ax.set_title(f"{mkt_name} {start_date}-{split_date}  TPs: {len(tpts1)}/{len(df1)}  {len(tpts1)/len(df1)*100:.2f}%")
    ax.legend()
    
    ax = axes[0, 1]
    ax.hist(res1['tdays'], bins=20, edgecolor='black')
    ax.set_title(f"DAYS {start_date}-{split_date}")
    
    ax = axes[0, 2]
    ax.hist(res1['tgap'], bins=20, edgecolor='black')
    ax.set_title(f"aGAP {start_date}-{split_date}")
    
    ax = axes[0, 3]
    ax.hist(res1['tr'], bins=20, edgecolor='black')
    ax.set_title(f"RETURN {start_date}-{split_date}")
    
    # 第二行：后半段
    ax = axes[1, 0]
    ax.plot(df2.index, df2['close'], label='Close')
    ax.scatter(tpts2['date'], tpts2['close'], color='red', s=30, label='Turning Points')
    ax.set_title(f"{mkt_name} {split_date}-{end_date}  TPs: {len(tpts2)}/{len(df2)}  {len(tpts2)/len(df2)*100:.2f}%")
    ax.legend()
    
    ax = axes[1, 1]
    ax.hist(res2['tdays'], bins=20, edgecolor='black')
    ax.set_title(f"DAYS {split_date}-{end_date}")
    
    ax = axes[1, 2]
    ax.hist(res2['tgap'], bins=20, edgecolor='black')
    ax.set_title(f"aGAP {split_date}-{end_date}")
    
    ax = axes[1, 3]
    ax.hist(res2['tmdd'], bins=20, edgecolor='black')
    ax.set_title(f"MDD {split_date}-{end_date}")
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{mkt_name}_{start_date}_{split_date}.png"))
    plt.close()

if __name__ == "__main__":
    # 示例：CSI300
    run_test('ZZ500', '000905.SH', '2002-01-04', '2019-12-31', '2026-02-25', './output')