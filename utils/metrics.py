import torch
import numpy as np

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    return None


def _flatten_result(obj):
    vals = []

    if obj is None:
        return vals

    arr = _to_numpy(obj)
    if arr is not None:
        vals.append(arr.reshape(-1))
        return vals

    if isinstance(obj, dict):
        for k in sorted(obj.keys()):
            vals.extend(_flatten_result(obj[k]))
        return vals

    if isinstance(obj, (list, tuple)):
        for item in obj:
            vals.extend(_flatten_result(item))
        return vals

    if isinstance(obj, (int, float, bool, np.number)):
        vals.append(np.array([obj], dtype=np.float32))
        return vals

    return vals


def compute_pcc(fp32_results, quant_results, eps=1e-12):
    fp32_flat = _flatten_result(fp32_results)
    quant_flat = _flatten_result(quant_results)

    if not fp32_flat or not quant_flat:
        return None, 0

    fp32_vec = np.concatenate(fp32_flat).astype(np.float64, copy=False)
    quant_vec = np.concatenate(quant_flat).astype(np.float64, copy=False)

    n = min(fp32_vec.size, quant_vec.size)
    if n == 0:
        return None, 0

    fp32_vec = fp32_vec[:n]
    quant_vec = quant_vec[:n]

    fp32_std = fp32_vec.std()
    quant_std = quant_vec.std()

    if fp32_std < eps or quant_std < eps:
        return None, n

    pcc = np.corrcoef(fp32_vec, quant_vec)[0, 1]
    return float(pcc), n