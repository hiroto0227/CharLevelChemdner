import torch
from torch.autograd import Variable


def get_variable(tensor, gpu=False, **kwargs):
    if torch.cuda.is_available() and gpu > 0:
        result = Variable(tensor.cuda(), **kwargs)
    else:
        result = Variable(tensor, **kwargs)
    return result


def checkpoint(model, model_path):
    print('\nmodel saved: {}'.format(model_path))
    torch.save(model.state_dict(), model_path)
