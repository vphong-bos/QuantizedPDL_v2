from aimet_torch import quantsim, bias_correction
from torch.utils.data import Dataset, DataLoader

def copy_biases(src_model, dst_model):
    src_modules = dict(src_model.named_modules())
    dst_modules = dict(dst_model.named_modules())

    copied = 0
    for name, src_module in src_modules.items():
        if name not in dst_modules:
            continue

        dst_module = dst_modules[name]

        src_bias = getattr(src_module, "bias", None)

        # Handle AIMET wrapped modules if needed
        if src_bias is None and hasattr(src_module, "_module_to_wrap"):
            src_bias = getattr(src_module._module_to_wrap, "bias", None)
        if dst_bias is None and hasattr(dst_module, "_module_to_wrap"):
            dst_bias = getattr(dst_module._module_to_wrap, "bias", None)

        if src_bias is not None and dst_bias is not None:
            dst_bias.data.copy_(src_bias.data)
            copied += 1

    print(f"Copied corrected biases for {copied} layers")

class BiasCorrectionDatasetWrapper(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]

        if isinstance(sample, dict):
            image = sample["image"]
        elif isinstance(sample, (list, tuple)):
            image = sample[0]
        else:
            image = sample

        return image, 0


def create_bias_correction_loader_from_calib_loader(calib_loader):
    return DataLoader(
        BiasCorrectionDatasetWrapper(calib_loader.dataset),
        batch_size=calib_loader.batch_size,
        shuffle=False,
        num_workers=calib_loader.num_workers,
        pin_memory=getattr(calib_loader, "pin_memory", False),
        drop_last=False,
    )

def apply_bias_correction(
    model,
    calib_loader,
    image_height,
    image_width,
    quant_scheme,
    default_param_bw,
    default_output_bw,
    config_file,
    bias_corr_num_quant_samples,
    bias_corr_num_bias_samples,
    bias_corr_empirical_only = True,

):
    model = model.cpu().eval()

    params = quantsim.QuantParams(
        weight_bw=default_param_bw,
        act_bw=default_output_bw,
        round_mode="nearest",
        quant_scheme=quant_scheme,
        config_file=config_file,
    )

    conv_bn_dict = None
    if not bias_corr_empirical_only:
        conv_bn_dict = bias_correction.find_all_conv_bn_with_activation(
            model,
            input_shape=(1, 3, image_height, image_width),
        )

    bc_loader = create_bias_correction_loader_from_calib_loader(calib_loader)

    bias_correction.correct_bias(
        model,
        params,
        num_quant_samples=min(bias_corr_num_quant_samples, len(calib_loader.dataset)),
        data_loader=bc_loader,
        num_bias_correct_samples=min(bias_corr_num_bias_samples, len(calib_loader.dataset)),
        conv_bn_dict=conv_bn_dict,
        perform_only_empirical_bias_corr=bias_corr_empirical_only,
    )

    return model