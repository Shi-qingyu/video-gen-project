import torch
import abc


LOW_RESOURCE = False


class AttentionStore():
    def __init__(self, num_steps: int, num_layers: int) -> None:
        self.num_steps = num_steps
        self.num_layers = num_layers
        self.attention_store = self.get_empty_store(num_layers)
        self.current_layer = 0
        self.current_step = 0
    
    def get_empty_store(self, num_layers: int):
        store = {}
        for i in range(num_layers):
            store[str(i)] = []
        return store

    def store(self, layer_idx: str, attention_map: torch.Tensor):
        if not isinstance(layer_idx, str):
            layer_idx = str(layer_idx)
        
        self.attention_store[layer_idx].append(attention_map)
        self.current_layer += 1
        if self.current_layer == self.num_layers:
            self.current_layer = 0
            self.current_step += 1
            if self.current_step == self.num_steps:
                self.integrate_attn_map()
    
    def integrate_attn_map(self):
        for key, value in self.attention_store.items(): 
            self.attention_store[key] = sum(value)  # (t, h, w, n)