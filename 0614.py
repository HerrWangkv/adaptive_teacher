import torch

model = torch.load("model_0009999.pth")
del model["optimizer"]
lst = []
for k in model['model'].keys():
    if "running_var" in k:
        lst.append(k[:-12])

new_model = type(model['model'])()
for k, v in model['model'].items():
    if '.'.join(k.split('.')[:4]) in lst:
        new_model['.'.join(k.split('.')[:4])+'.main_bn.'+k.split('.')[-1]] = v
        new_model['.'.join(k.split('.')[:4])+'.aux_bn.'+k.split('.')[-1]] = torch.zeros_like(v)
    else:
        new_model[k] = v
model['model'] = new_model
torch.save(model, "model_0009999_wo_optimizer.pth")
