import torch
import numpy as np
import time
from ptflops import get_model_complexity_info
from sklearn import metrics
from operator import itemgetter

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
    torch.cuda.reset_peak_memory_stats()
    # empty_gpu = torch.cuda.memory_stats(model.device)['active.all.current']

    for i, bs in enumerate(batch_sizes):
        with torch.no_grad():
            inp = torch.randn(bs, *input_size).cuda(non_blocking=False)
            _ = model(inp)
            # current_memory = torch.cuda.memory_stats(
                # model.device)['active.all.current']
            # memory.append(model_memory + current_memory-empty_gpu)
            current_memory = torch.cuda.memory_stats(
                model.device)['allocated_bytes.all.peak'] / (1024**3)
            memory.append(current_memory)
            torch.cuda.reset_peak_memory_stats()
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

    torch.cuda.empty_cache()

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


def input_constructor(input_res):
    return {'x': torch.randn(*input_res).cuda()}

def compute_complexity(model, input_size, batch_size=1):
    macs, params = get_model_complexity_info(model, (batch_size, *input_size),
            input_constructor=input_constructor, as_strings=True, verbose=False, print_per_layer_stat=False)

    return macs


def compute_parameters(model):
    total_params = 0
    trainable_params = 0

    for name, parameter in model.named_parameters():
        param = parameter.numel()
        total_params += param
        if parameter.requires_grad:
            trainable_params += param
    return total_params, trainable_params


def tuneThresholdfromScore(scores, labels, target_fa, target_fr=None):

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = np.nanargmin(np.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])

    for tfa in target_fa:
        # numpy.where(fpr<=tfa)[0][-1]
        idx = np.nanargmin(np.absolute((tfa - fpr)))
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])

    idxE = np.nanargmin(np.absolute((fnr - fpr)))
    eer = max(fpr[idxE], fnr[idxE])*100

    return (tunedThreshold, eer, fpr, fnr)

# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.


def ComputeErrorRates(scores, labels):

    # Sort the scores from smallest to largest, and also get the corresponding
    # indexes of the sorted scores.  We will treat the sorted scores as the
    # thresholds at which the the error-rates are evaluated.
    sorted_indexes, thresholds = zip(*sorted(
        [(index, threshold) for index, threshold in enumerate(scores)],
        key=itemgetter(1)))
    sorted_labels = []
    labels = [labels[i] for i in sorted_indexes]
    fnrs = []
    fprs = []

    # At the end of this loop, fnrs[i] is the number of errors made by
    # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
    # is the total number of times that we have correctly accepted scores
    # greater than thresholds[i].
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i-1] + labels[i])
            fprs.append(fprs[i-1] + 1 - labels[i])
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm

    # Now divide by the total number of false negative errors to
    # obtain the false positive rates across all thresholds
    fnrs = [x / float(fnrs_norm) for x in fnrs]

    # Divide by the total number of corret positives to get the
    # true positive rate.  Subtract these quantities from 1 to
    # get the false positive rates.
    fprs = [1 - x / float(fprs_norm) for x in fprs]
    return fnrs, fprs, thresholds

# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.


def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold
