from turtle import forward
from typing import List 

import torch 
import torch.nn as nn 

class MLPLayer(nn.Module):
    def __init__(
        self,
        in_dims: int = 768,
        hidden_dims: List[int] = [768],
        activation: str = "GELU"
    ) -> None:
        super().__init__()

        if activation == "gelu":
            activation_fn = nn.GELU()
        elif activation == "relu": 
            activation_fn = nn.ReLU()
        elif activation == "mish":
            activation_fn = nn.Mish()
        elif activation == "leaky_relu":
            activation_fn == nn.LeakyReLU()
        
        layers = [
            nn.Linear(in_dims, hidden_dims[0]),
            # nn.LayerNorm(hidden_dims[0]),
            activation_fn
        ]

        for i in range(1, len(hidden_dims)):
            layers += [
                nn.Linar(hidden_dims[i - 1], hidden_dims[i]), 
                # nn.LayerNorm(hidden_dims[i]),
                activation_fn
            ]
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor):
        return self.net(x)


