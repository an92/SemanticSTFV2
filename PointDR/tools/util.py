import os
import time

from torchpack.environ import set_run_dir
from torchpack.utils.config import configs


def auto_time_set_run_dir() -> str:
    tags = ['run']

    # 获取当前时间戳（秒级），并转换为 8 位字符串
    timestamp = int(time.time())  # 获取当前时间的秒级时间戳
    time_str = str(timestamp)[-8:]  # 取时间戳的后 8 位

    # 修改 run_dir，加入基于时间的 8 位标签
    run_dir = os.path.join('runs', '-'.join(tags) + f"_{time_str}")
    set_run_dir(run_dir)
    return run_dir