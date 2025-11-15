import torch
torch.cuda.empty_cache()  # 清理 GPU 缓存
import gc
gc.collect()  # 清理 Python 垃圾回收