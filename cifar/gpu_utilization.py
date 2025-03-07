import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
import pynvml
import os

class MultiGPUUtilizationLogger(Callback):
    def __init__(self, log_frequency=100):
        super().__init__()
        self.log_frequency = log_frequency
        # Initialize NVIDIA Management Library
        pynvml.nvmlInit()
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_frequency == 0:
            # Only log from rank 0 process on each node to avoid duplication
            if trainer.local_rank == 0:
                # Get number of GPUs visible to this process
                device_count = pynvml.nvmlDeviceGetCount()
                node_id = os.environ.get('NODE_RANK', 0)
                
                for i in range(device_count):
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        
                        # Log to TensorBoard with node and GPU identifiers
                        trainer.logger.experiment.add_scalar(
                            f"gpu/node{node_id}_gpu{i}/utilization", 
                            utilization.gpu, 
                            trainer.global_step
                        )
                        trainer.logger.experiment.add_scalar(
                            f"gpu/node{node_id}_gpu{i}/memory_used_percent", 
                            memory_info.used / memory_info.total * 100, 
                            trainer.global_step
                        )
                    except Exception as e:
                        print(f"Error logging GPU {i} stats: {e}")