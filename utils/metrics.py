import torch
import numpy as np
import time
import GPUtil
from utils.complexity_utils import add_flops_counting_methods, summary
import copy

class AverageMeter:
    """
    Class to be an average meter for any average metric like loss, accuracy, etc.
    """

    def __init__(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def val(self):
        return self.avg


class AverageMeterList:
    """
    Class to be an average meter for any average metric List structure
    """

    def __init__(self, num_cls):
        self.cls = num_cls
        self.value = [0] * self.cls
        self.avg = [0] * self.cls
        self.sum = [0] * self.cls
        self.count = [0] * self.cls
        self.reset()

    def reset(self):
        self.value = [0] * self.cls
        self.avg = [0] * self.cls
        self.sum = [0] * self.cls
        self.count = [0] * self.cls

    def update(self, val, n=1):
        for i in range(self.cls):
            self.value[i] = val[i]
            self.sum[i] += val[i] * n
            self.count[i] += n
            self.avg[i] = self.sum[i] / self.count[i]

    @property
    def val(self):
        return self.avg


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def compute_memory_usage(model, input_size, batch_sizes=[1, 2, 4, 8, 16, 32, 64], model_memory=0):
    memory = []
    gpu = GPUtil.getGPUs()[0]
    empty_gpu = gpu.memoryUsed

    for i, bs in enumerate(batch_sizes):
        with torch.no_grad():
            _ = model(torch.randn(
                bs, *input_size).cuda(non_blocking=True))

        memory.append(model_memory + gpu.memoryUsed-empty_gpu)
    torch.cuda.empty_cache()

    return memory


def measure(model, x):
	# synchronize gpu time and measure fp
	torch.cuda.synchronize()
	t0 = time.time()
	with torch.no_grad():
		y_pred = model(x)
	torch.cuda.synchronize()
	elapsed_fp = time.time() - t0

	return elapsed_fp


def benchmark(model, x):
	# transfer the model on GPU
	model = model.cuda().eval()

	# DRY RUNS
	for i in range(10):
		_ = measure(model, x)

	# print('DONE WITH DRY RUNS, NOW BENCHMARKING')

	# START BENCHMARKING
	t_forward = []
	t_backward = []
	for i in range(10):
		t_fp = measure(model, x)
		t_forward.append(t_fp)

	# free memory
	del model

	return t_forward

def compute_inference_time(model, input_size, batch_sizes=[1, 2, 4, 8, 16, 32, 64]):
    mean_tfp = []
    std_tfp = []
    for i, bs in enumerate(batch_sizes):
        x = torch.randn(bs, *input_size).cuda()
        tmp = benchmark(model, x)
        # NOTE: we are estimating inference time per sample
        mean_tfp.append(np.asarray(tmp).mean() / bs*1e3)
        std_tfp.append(np.asarray(tmp).std() / bs*1e3)

    return mean_tfp, std_tfp

def compute_complexity(model, input_size, batch_size=64):
    model = model.cuda().eval()
    model = add_flops_counting_methods(model)
    model.start_flops_count()

    with torch.no_grad():
        x = torch.randn(batch_size, *input_size).cuda(non_blocking=True)
        _ = model(x)

    return model.compute_average_flops_cost() / 1e9 / 2


def compute_parameters(model):
    total_params = 0
    trainable_params = 0

    for name, parameter in model.named_parameters():
        param = parameter.numel()
        total_params += param
        if parameter.requires_grad:
            trainable_params += param
    return total_params, trainable_params
