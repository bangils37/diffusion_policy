from typing import Optional, Dict
import os

class TopKCheckpointManager:
    def __init__(self,
            save_dir,
            monitor_key: str,
            mode='min',
            k=1,
            format_str='epoch={epoch:04d}-val_loss={val_loss:.4f}.ckpt'
        ):
        assert mode in ['max', 'min']
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = dict()
    
    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None

        if self.monitor_key not in data:
            return None

        value = data[self.monitor_key]
        if value is None:
            return None
        try:
            value = float(value)
        except (ValueError, TypeError):
            return None
        if value != value:  # NaN check
            return None

        ckpt_path = os.path.join(
            self.save_dir, self.format_str.format(**data))
        
        if len(self.path_value_map) < self.k:
            # under-capacity
            self.path_value_map[ckpt_path] = value
            return ckpt_path
        
        # at capacity
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == 'max':
            if value > min_value:
                delete_path = min_path
        else:
            if value < max_value:
                delete_path = max_path

        if delete_path is None:
            return None
        else:
            del self.path_value_map[delete_path]
            self.path_value_map[ckpt_path] = value

            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir, exist_ok=True)

            if os.path.exists(delete_path) and delete_path != ckpt_path:
                try:
                    os.remove(delete_path)
                except OSError:
                    pass
            return ckpt_path


class RecentCheckpointManager:
    """
    Maintains the most recent K checkpoints (FIFO queue).
    When total checkpoints exceed K, the oldest checkpoint is removed.
    """
    def __init__(self,
            save_dir,
            k=5,
            format_str='epoch={epoch:04d}.ckpt'
        ):
        assert k >= 0
        self.save_dir = save_dir
        self.k = k
        self.format_str = format_str
        self.history_paths = []
        
        # Discover any pre-existing checkpoints matching format in save_dir
        if os.path.exists(save_dir) and os.path.isdir(save_dir):
            prefix = format_str.split('{')[0]
            suffix = format_str.split('}')[-1]
            existing = [
                os.path.join(save_dir, f)
                for f in os.listdir(save_dir)
                if f.startswith(prefix) and f.endswith(suffix) and f != 'latest.ckpt' and 'val_loss' not in f
            ]
            # Sort chronologically by file mtime
            existing.sort(key=lambda x: os.path.getmtime(x))
            self.history_paths = existing
    
    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None

        ckpt_path = os.path.join(
            self.save_dir, self.format_str.format(**data))
        
        if ckpt_path in self.history_paths:
            self.history_paths.remove(ckpt_path)
        self.history_paths.append(ckpt_path)
        
        while len(self.history_paths) > self.k:
            oldest_path = self.history_paths.pop(0)
            if os.path.exists(oldest_path) and oldest_path != ckpt_path:
                try:
                    os.remove(oldest_path)
                except OSError:
                    pass

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
            
        return ckpt_path


