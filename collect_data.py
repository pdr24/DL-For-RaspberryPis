import pandas as pd
import numpy as np
import psutil

# returns cpu usage %, load avgs over 1/5/15 mins, and current cpu frequency
def collect_cpu_statistics():
    cpu_percent = psutil.cpu_percent(0.1)

    cpu_load_avg = psutil.getloadavg()
    load_1min = cpu_load_avg[0]
    load_5min = cpu_load_avg[1]
    load_15min = cpu_load_avg[2]

    cpu_freq = psutil.cpu_freq()
    cpu_freq_current = cpu_freq.current

    return cpu_percent, load_1min, load_5min, load_15min, cpu_freq_current

# returns various memory related metrics 
def collect_memory_statistics():
    virtual_memory = psutil.virtual_memory()
    available_memory = virtual_memory.available
    used_memory = virtual_memory.used
    percent_used = virtual_memory.percent
    active_memory = virtual_memory.active
    inactive_memory = virtual_memory.inactive
    buffers = virtual_memory.buffers
    cached = virtual_memory.cached
    shared = virtual_memory.shared

    swap_memory = psutil.swap_memory()
    swap_used = swap_memory.used
    swap_free = swap_memory.free
    swap_percent = swap_memory.percent

    return available_memory, used_memory, percent_used, active_memory, inactive_memory, buffers, cached, shared, swap_used, swap_free, swap_percent

# this function adds all of the data to a new row in a dataframe
def add_metrics_to_df(df, inference_time, cpu_percent, load_1min, load_5min, load_15min, cpu_freq_current, available_memory, used_memory, percent_used, active_memory, inactive_memory, buffers, cached, shared, swap_used, swap_free, swap_percent):
    new_row = pd.DataFrame([[inference_time, cpu_percent, load_1min, load_5min, load_15min, cpu_freq_current, available_memory, used_memory, percent_used, active_memory, inactive_memory, buffers, cached, shared, swap_used, swap_free, swap_percent]], columns=df.columns)
    return pd.concat([df, new_row], ignore_index=True)

def setup_df():
    system_metrics = pd.DataFrame(columns=[
        "Inference Time", "CPU Usage Percent", "CPU Load 1 min", "CPU Load 5 min", "CPU Load 15 min", "CPU Freq (current)",
        "Available Memory", "Used Memory", "Percent Memory Used", "Active Memory", "Inactive Memory", "Buffers", 
        "Cached", "Shared Memory", "Swap Used", "Swap Free", "Swap Percent"
        ])
    return system_metrics