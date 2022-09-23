import torch 
import torch.nn as nn 


class MultiTaskLossWrapper(nn.Module):
    def __init__(
        self, 
        num_tasks
    ) -> None:

        super().__init__()
        self.num_tasks = num_tasks
        
        self.log_vars = nn.Parameter(torch.zeros((num_tasks)))
    
    def forward(self, *args):
        """
            args: Loss outputs of different loss functions for different tasks 
        """

        L = torch.exp(-self.log_vars[0]) * args[0] + self.log_vars[0]

        for i in range(1, len(args)):
            l = torch.exp(-self.log_vars[i]) * args[i] + self.log_vars[i]
            L += l 
        
        return L 
