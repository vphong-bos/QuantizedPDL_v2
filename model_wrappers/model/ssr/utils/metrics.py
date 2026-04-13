import mmcv
import torch
import warnings

from model_wrappers.model.ssr.utils.dataset import extract_data

warnings.filterwarnings("ignore")

from model_wrappers.model.ssr.mmdet3d_plugin.SSR.planner.metric_stp3 import PlanningMetric


def get_model_result(model_obj, data):
    model = model_obj["model"]
    is_quant = model_obj.get("is_quant", 0)
    with torch.no_grad():
        if is_quant:
            result = model(return_loss=False, rescale=True, data=data)

        if not is_quant:
            result = model(return_loss=False, rescale=True, **data)
    return result


def move_data_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device, non_blocking=True)

    if isinstance(data, dict):
        return {k: move_data_to_device(v, device) for k, v in data.items()}

    if isinstance(data, list):
        return [move_data_to_device(v, device) for v in data]

    if isinstance(data, tuple):
        return tuple(move_data_to_device(v, device) for v in data)

    return data


def evaluate_model(
    model_obj,
    data_loader,
    max_samples=20,
    device=None,
):
    if device is None:
        if isinstance(model_obj, dict):
            device = model_obj.get("device", "cpu")
        else:
            device = "cpu"

    if isinstance(device, torch.device):
        device = str(device)

    if isinstance(model_obj, dict):
        backend = model_obj.get("backend", "torch")
        model = model_obj.get("model", None)
        normalized_model_obj = dict(model_obj)
    else:
        backend = getattr(model_obj, "backend", "torch")
        model = model_obj
        normalized_model_obj = {
            "backend": backend,
            "model": model,
        }

    normalized_model_obj["device"] = device

    if backend == "torch" and model is not None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested torch device '{device}' but this PyTorch build has no CUDA support."
            )
        model.to(device)
        model.eval()

    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    try:
        for i, data in enumerate(data_loader):
            if max_samples is not None and i >= max_samples:
                break

            data = extract_data(data)

            if backend == "torch":
                data = move_data_to_device(data, device)

            result = get_model_result(normalized_model_obj, data)

            results.extend(result)

            batch_size = len(result)
            for _ in range(batch_size):
                prog_bar.update()

    except KeyboardInterrupt:
        print("Keyboard interrupt, exiting...")

    return results