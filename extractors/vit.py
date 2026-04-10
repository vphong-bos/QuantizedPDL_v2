# Missing input qparams (25):
#   - encoder_layers.0.attention.output.dense
#   - encoder_layers.0.layernorm_after
#   - encoder_layers.0.layernorm_before
#   - encoder_layers.1.attention.output.dense
#   - encoder_layers.1.layernorm_after
#   - encoder_layers.1.layernorm_before
#   - encoder_layers.10.attention.output.dense
#   - encoder_layers.11.attention.output.dense
#   - encoder_layers.11.layernorm_after
#   - encoder_layers.2.attention.output.dense
#   - encoder_layers.2.layernorm_after
#   - encoder_layers.2.layernorm_before
#   - encoder_layers.3.attention.output.dense
#   - encoder_layers.3.layernorm_after
#   - encoder_layers.3.layernorm_before
#   - encoder_layers.4.attention.output.dense
#   - encoder_layers.4.layernorm_after
#   - encoder_layers.4.layernorm_before
#   - encoder_layers.5.attention.output.dense
#   - encoder_layers.5.layernorm_after
#   - encoder_layers.5.layernorm_before
#   - encoder_layers.6.attention.output.dense
#   - encoder_layers.7.attention.output.dense
#   - encoder_layers.8.attention.output.dense
#   - encoder_layers.9.attention.output.dense
# Missing output qparams (13):
#   - encoder_layers.0.layernorm_after
#   - encoder_layers.0.layernorm_before
#   - encoder_layers.1.layernorm_after
#   - encoder_layers.1.layernorm_before
#   - encoder_layers.11.layernorm_after
#   - encoder_layers.2.layernorm_after
#   - encoder_layers.2.layernorm_before
#   - encoder_layers.3.layernorm_after
#   - encoder_layers.3.layernorm_before
#   - encoder_layers.4.layernorm_after
#   - encoder_layers.4.layernorm_before
#   - encoder_layers.5.layernorm_after
#   - encoder_layers.5.layernorm_before

from .base import QuantizedOnnxExtractor


class ViTExtractor(QuantizedOnnxExtractor):
    def __init__(self, ckpt_path):
        # Extend passthrough ops for ViT's transformer architecture
        super().__init__(ckpt_path)
        
        # Add more passthrough ops for ViT
        self.passthrough_ops = self.passthrough_ops | {
            "Mul", "Add", "Sub", "Div",  # Arithmetic
            "Tile", "Expand", "Concat",   # Reshaping
            "Constant", "ConstantOfShape",  # Constants
            "Where", "RoiAlign", "Pad", "Slice", "Resize",  # Other
            "ReduceMean", "ReduceSum", "Shape", "Gather",  # Reductions
        }

    def _find_compute_node_from_weight_qdq(self, weight_tensor_name):
        """Override to include LayerNormalization as compute op for ViT."""
        queue = [weight_tensor_name]
        seen = set()
        vit_compute_ops = self.COMPUTE_OPS | {"LayerNormalization"}
        
        while queue:
            tensor = queue.pop(0)
            if tensor in seen:
                continue
            seen.add(tensor)
            for consumer in self.consumers.get(tensor, []):
                if consumer.op_type in vit_compute_ops:
                    return consumer
                if consumer.op_type in self.passthrough_ops:
                    queue.extend(consumer.output)
        return None

    def _get_activation_roles(self, export_prefix):
        """ViT-specific role assignment.
        
        - Layernorm layers: output only
        - Attention output dense: output only (similar to SSR)
        - Everything else: input + output
        """
        prefix_lower = export_prefix.lower()
        
        if "layernorm" in prefix_lower:
            return {"output"}
        
        if "attention.output.dense" in prefix_lower:
            return {"output"}
        
        return {"input", "output"}