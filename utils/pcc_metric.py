import torch

def extract_input(batch):
    if torch.is_tensor(batch):
        return batch

    if isinstance(batch, dict):
        for key in ["image", "images", "input", "inputs"]:
            if key in batch:
                return extract_input(batch[key])

        # fallback: tìm tensor đầu tiên trong dict
        for v in batch.values():
            try:
                return extract_input(v)
            except Exception:
                pass

        raise KeyError(f"Batch dict does not contain supported keys. Got: {list(batch.keys())}")

    if isinstance(batch, (list, tuple)):
        if len(batch) == 0:
            raise ValueError("Empty batch list/tuple.")

        # ưu tiên tensor ảnh BCHW
        for item in batch:
            if torch.is_tensor(item) and item.ndim == 4:
                return item

        # fallback: recurse từng item
        for item in batch:
            try:
                return extract_input(item)
            except Exception:
                pass

        raise TypeError("Could not extract image tensor from batch list/tuple.")

    raise TypeError(f"Unsupported batch type: {type(batch)}")

def extract_tensor(output):
    """
    Tries to extract the main prediction tensor from common output formats.
    """
    if torch.is_tensor(output):
        return output

    if isinstance(output, dict):
        for key in ["semantic", "sem_logits", "logits", "out", "pred"]:
            if key in output and torch.is_tensor(output[key]):
                return output[key]
        for v in output.values():
            if torch.is_tensor(v):
                return v

    if isinstance(output, (list, tuple)):
        for v in output:
            if torch.is_tensor(v):
                return v

    raise TypeError(f"Could not extract tensor from model output type: {type(output)}")


def pearson_corrcoef(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.float().reshape(-1)
    y = y.float().reshape(-1)

    if torch.isnan(x).any() or torch.isnan(y).any():
        print("found nan")
        return torch.tensor(0.0, device=x.device)

    x = x - x.mean()
    y = y - y.mean()

    denom = torch.sqrt((x * x).sum()) * torch.sqrt((y * y).sum())
    if torch.isnan(denom) or denom.abs() < eps:
        return torch.tensor(0.0, device=x.device)

    corr = (x * y).sum() / (denom + eps)
    if torch.isnan(corr):
        print("found nan")
        return torch.tensor(0.0, device=x.device)

    return corr


@torch.no_grad()
def evaluate_pcc(fp32_model, quant_model, loader, device, max_samples=-1):
    fp32_model.eval()
    quant_model.eval()

    pcc_values = []
    seen = 0

    for batch in loader:
        inputs = extract_input(batch)

        if not torch.is_tensor(inputs):
            raise TypeError(f"Extracted input is not a tensor: {type(inputs)}")

        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(0)

        if inputs.ndim != 4:
            raise ValueError(f"Expected BCHW input, got shape {tuple(inputs.shape)}")

        inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)

        fp32_out = extract_tensor(fp32_model(inputs))
        quant_out = extract_tensor(quant_model(inputs))

        if fp32_out.shape != quant_out.shape:
            raise ValueError(
                f"FP32 and quantized outputs have different shapes: "
                f"{tuple(fp32_out.shape)} vs {tuple(quant_out.shape)}"
            )

        batch_size = fp32_out.shape[0]
        for i in range(batch_size):
            pcc = pearson_corrcoef(fp32_out[i], quant_out[i])
            pcc_values.append(pcc.item())
            seen += 1

            if max_samples > 0 and seen >= max_samples:
                return {"PCC": sum(pcc_values) / len(pcc_values)}

    return {"PCC": sum(pcc_values) / len(pcc_values) if pcc_values else 0.0}