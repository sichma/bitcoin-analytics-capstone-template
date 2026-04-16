# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:01:42 2026

@author: GASH000
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import akshare as ak
import time

# ---------- 配置 ----------
# 当前日期
today = pd.Timestamp.now().normalize()
# 时间范围：过去6个月到未来12个月
start_date = today - pd.DateOffset(months=6)
end_date = today + pd.DateOffset(months=12)

# ---------- 日期生成函数（规则生成，同前）----------
def get_nfp_dates(months):
    dates = []
    for month_start in months:
        first_friday = month_start + pd.offsets.Week(weekday=4)
        if first_friday <= month_start + pd.offsets.MonthEnd():
            dates.append(first_friday)
    return dates

def get_jolts_dates(months):
    dates = []
    for month_start in months:
        first_tuesday = month_start + pd.offsets.Week(weekday=1)
        if first_tuesday <= month_start + pd.offsets.MonthEnd():
            dates.append(first_tuesday)
    return dates

def get_cpi_dates(months):
    dates = []
    for month_start in months:
        raw_date = month_start.replace(day=13)
        if raw_date.weekday() >= 5:
            raw_date = raw_date + pd.offsets.Week(weekday=0)
        dates.append(raw_date)
    return dates

def get_pce_dates(months):
    dates = []
    for month_start in months:
        month_end = month_start + pd.offsets.MonthEnd()
        last_bday = month_end
        while last_bday.weekday() >= 5:
            last_bday -= pd.Timedelta(days=1)
        dates.append(last_bday)
    return dates

# FOMC日程（2024-2026，手动维护）
fomc_dates = [
    pd.Timestamp('2024-01-31'), pd.Timestamp('2024-03-20'), pd.Timestamp('2024-05-01'),
    pd.Timestamp('2024-06-12'), pd.Timestamp('2024-07-31'), pd.Timestamp('2024-09-18'),
    pd.Timestamp('2024-11-07'), pd.Timestamp('2024-12-18'),
    pd.Timestamp('2025-01-29'), pd.Timestamp('2025-03-19'), pd.Timestamp('2025-05-07'),
    pd.Timestamp('2025-06-18'), pd.Timestamp('2025-07-30'), pd.Timestamp('2025-09-17'),
    pd.Timestamp('2025-10-29'), pd.Timestamp('2025-12-10'),
    pd.Timestamp('2026-01-28'), pd.Timestamp('2026-03-18'),  # 可继续添加
]

# 生成月度序列
months = pd.date_range(start=start_date, end=end_date, freq='MS')
nfp_dates = get_nfp_dates(months)
jolts_dates = get_jolts_dates(months)
cpi_dates = get_cpi_dates(months)
pce_dates = get_pce_dates(months)
fomc_dates_in_range = [d for d in fomc_dates if start_date <= d <= end_date]

# ---------- akshare 数据获取函数 ----------
def get_us_cpi(date):
    """获取美国CPI（月度）"""
    try:
        # akshare 美国CPI数据
        df = ak.macro_usa_cpi()
        if df.empty:
            return "获取失败"
        # df通常包含 '日期' 和 '今值' 列，转换日期格式
        df['日期'] = pd.to_datetime(df['日期'])
        # 取发布日前一个月的数据（CPI发布的是上月数据）
        target_month = date - pd.DateOffset(months=1)
        # 找到最接近的月份
        df['diff'] = abs(df['日期'] - target_month)
        closest = df.loc[df['diff'].idxmin()]
        cpi_value = closest['今值']
        return f"CPI: {cpi_value:.1f} (同比%)"
    except Exception as e:
        print(f"CPI获取失败: {e}")
        return "待获取"

def get_us_pce(date):
    """获取美国PCE（月度）"""
    try:
        # akshare 美国PCE数据（可能接口名：macro_usa_pce）
        # 注意：akshare 中可能有 macro_usa_pce，若无则尝试其他
        df = ak.macro_usa_pce()
        if df.empty:
            # 尝试备用接口
            df = ak.macro_usa_core_pce()
        if df.empty:
            return "获取失败"
        df['日期'] = pd.to_datetime(df['日期'])
        target_month = date - pd.DateOffset(months=1)
        df['diff'] = abs(df['日期'] - target_month)
        closest = df.loc[df['diff'].idxmin()]
        pce_value = closest['今值']
        return f"PCE: {pce_value:.2f} (同比%)"
    except Exception as e:
        print(f"PCE获取失败: {e}")
        return "待获取"

def get_us_nfp(date):
    """获取美国非农就业人数"""
    try:
        # akshare 美国非农数据
        df = ak.macro_usa_non_farm()
        if df.empty:
            return "获取失败"
        df['日期'] = pd.to_datetime(df['日期'])
        target_month = date - pd.DateOffset(months=1)
        df['diff'] = abs(df['日期'] - target_month)
        closest = df.loc[df['diff'].idxmin()]
        nfp_value = closest['今值']  # 通常单位为千人
        return f"非农新增: {nfp_value:.0f}K"
    except Exception as e:
        print(f"非农获取失败: {e}")
        return "待获取"

def get_us_jolts(date):
    """获取美国JOLTS职位空缺"""
    try:
        df = ak.macro_usa_jolts()
        if df.empty:
            return "获取失败"
        df['日期'] = pd.to_datetime(df['日期'])
        target_month = date - pd.DateOffset(months=1)
        df['diff'] = abs(df['日期'] - target_month)
        closest = df.loc[df['diff'].idxmin()]
        jolts_value = closest['今值']
        return f"职位空缺: {jolts_value:.0f}K"
    except Exception as e:
        print(f"JOLTS获取失败: {e}")
        return "待获取"

def get_fed_rate(date):
    """获取FOMC利率（联邦基金利率）"""
    try:
        # akshare 美国联邦基金利率
        df = ak.macro_usa_interest_rate()
        if df.empty:
            return "获取失败"
        df['日期'] = pd.to_datetime(df['日期'])
        # 找会议日附近的数据
        df['diff'] = abs(df['日期'] - date)
        closest = df.loc[df['diff'].idxmin()]
        rate = closest['今值']
        return f"利率: {rate:.2f}%"
    except Exception as e:
        print(f"利率获取失败: {e}")
        return "待获取"

# ---------- 构建事件列表 ----------
events = []

# 非农
for d in nfp_dates:
    details = get_us_nfp(d) if d <= today else "待发布"
    events.append({
        '事件名称': '非农就业报告 (含时薪增速)',
        '发布日期': d,
        '详细内容': details,
        '数据来源': '美国劳工统计局'
    })

# JOLTS
for d in jolts_dates:
    details = get_us_jolts(d) if d <= today else "待发布"
    events.append({
        '事件名称': 'JOLTS职位空缺',
        '发布日期': d,
        '详细内容': details,
        '数据来源': '美国劳工统计局'
    })

# CPI
for d in cpi_dates:
    details = get_us_cpi(d) if d <= today else "待发布"
    events.append({
        '事件名称': 'CPI & 核心CPI',
        '发布日期': d,
        '详细内容': details,
        '数据来源': '美国劳工统计局'
    })

# PCE
for d in pce_dates:
    details = get_us_pce(d) if d <= today else "待发布"
    events.append({
        '事件名称': 'PCE & 核心PCE',
        '发布日期': d,
        '详细内容': details,
        '数据来源': '美国经济分析局'
    })

# FOMC
for d in fomc_dates_in_range:
    details = get_fed_rate(d) if d <= today else "待发布"
    events.append({
        '事件名称': 'FOMC利率决议',
        '发布日期': d,
        '详细内容': details,
        '数据来源': '美联储'
    })

# 转换为DataFrame
df_events = pd.DataFrame(events)
df_events.drop_duplicates(subset=['事件名称', '发布日期'], inplace=True)
df_events.sort_values('发布日期', inplace=True)
df_events.reset_index(drop=True, inplace=True)
df_events['距今天数'] = (df_events['发布日期'] - today).dt.days
df_events['状态'] = df_events['发布日期'].apply(lambda x: '未来' if x >= today else '过去')

# 显示
print(f"当前日期: {today.strftime('%Y-%m-%d')}")
print(f"显示范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
df_events