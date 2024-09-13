import torch.nn as nn

def polyak_averaging(net: nn.Module, target_net: nn.Module, tau: float):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_((target_param.data * (1.0 - tau)) + (param.data * tau))


def bn_running_stats_polyak_averaging(net: nn.Module, target_net: nn.Module, tau: float):
    for module, target_module in zip(net.modules(), target_net.modules()):
         for key in ['running_mean', 'running_var']:
              if hasattr(module, key) and hasattr(target_module, key):
                   param = getattr(module, key)
                   target_param = getattr(target_module, key)
                   target_param.data.copy_((target_param.data * (1.0 - tau)) + (param.data * tau))