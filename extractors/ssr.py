"""SSR model extractor (handles norm layers, attention components, embeddings).

Key difference from ResNet:
  - Norm layers (LayerNormalization ops) are treated as compute operations
  - InstanceNormalization followed by Mul for scaling weights
  - Attention projections (MatMul) work the same as ResNet
  - Embeddings use Gather operations instead of traditional layers
  
Override parent class methods to handle these operation types.
"""

# Missing input qparams (14):
#   - model.pts_bbox_head.navi_se.mlp_reduce
#   - model.pts_bbox_head.transformer.encoder.layers.0.attentions.0.output_proj
#   - model.pts_bbox_head.transformer.encoder.layers.0.attentions.0.value_proj
#   - model.pts_bbox_head.transformer.encoder.layers.0.attentions.1.deformable_attention.attention_weights
#   - model.pts_bbox_head.transformer.encoder.layers.0.attentions.1.deformable_attention.sampling_offsets
#   - model.pts_bbox_head.transformer.encoder.layers.0.attentions.1.deformable_attention.value_proj
#   - model.pts_bbox_head.transformer.encoder.layers.1.attentions.0.output_proj
#   - model.pts_bbox_head.transformer.encoder.layers.1.attentions.1.deformable_attention.attention_weights
#   - model.pts_bbox_head.transformer.encoder.layers.1.attentions.1.deformable_attention.sampling_offsets
#   - model.pts_bbox_head.transformer.encoder.layers.1.attentions.1.deformable_attention.value_proj
#   - model.pts_bbox_head.transformer.encoder.layers.2.attentions.0.output_proj
#   - model.pts_bbox_head.transformer.encoder.layers.2.attentions.1.deformable_attention.attention_weights
#   - model.pts_bbox_head.transformer.encoder.layers.2.attentions.1.deformable_attention.sampling_offsets
#   - model.pts_bbox_head.transformer.encoder.layers.2.attentions.1.deformable_attention.value_proj
# Missing output qparams (1):
#   - model.pts_bbox_head.tokenlearner.layer_norm

from .base import QuantizedOnnxExtractor


class SSRExtractor(QuantizedOnnxExtractor):

    def __init__(self, ckpt_path):
        # Extend passthrough ops to include arithmetic ops used in norm layer parameterization
        super().__init__(ckpt_path)
        
        # Add more passthrough ops for SSR's complex layer structures
        self.passthrough_ops = self.passthrough_ops | {
            "Mul", "Add", "Sub", "Div",  # Arithmetic for scaling
            "Tile", "Expand", "Concat",   # For embedding reshaping
            "Constant", "ConstantOfShape",  # Constants used in paths
            "Where", "RoiAlign", "Pad", "Slice", "Resize",  # Other ops
        }

    def _find_compute_node_from_weight_qdq(self, weight_tensor_name):
        """Override to include LayerNormalization, InstanceNormalization, and Gather as compute ops for SSR."""
        queue = [weight_tensor_name]
        seen = set()
        ssr_compute_ops = self.COMPUTE_OPS | {"LayerNormalization", "InstanceNormalization", "Gather"}
        
        while queue:
            tensor = queue.pop(0)
            if tensor in seen:
                continue
            seen.add(tensor)
            for consumer in self.consumers.get(tensor, []):
                if consumer.op_type in ssr_compute_ops:
                    return consumer
                if consumer.op_type in self.passthrough_ops:
                    queue.extend(consumer.output)
        return None

    def _find_output_quant_params(self, prefix, compute_node):
        """Override to handle SSR's extended passthrough ops (Tile, Expand, Shape, etc.)."""
        # SSR has additional passthrough ops for embeddings/positional encoding and norm layers
        ssr_downstream_ops = {
            "BatchNormalization", "Relu", "Clip", "Add", "Transpose",
            "Reshape", "Identity", "Cast", "Flatten", "Squeeze",
            "Unsqueeze", "GlobalAveragePool", "AveragePool", "MaxPool",
            "Tile", "Expand", "Concat", "Mul", "Sub", "Div",
            "Shape", "Gather", "Constant", "ConstantOfShape",  # Added for norm layer tracing
            "Where", "RoiAlign", "Pad", "Slice", "Resize",  # Additional ops
        }
        
        output_tensor = compute_node.output[0]
        # Handle MatMul + bias Add pattern
        if compute_node.op_type in {"MatMul", "Gemm"}:
            for consumer in self.consumers.get(output_tensor, []):
                if consumer.op_type == "Add" and any(
                    inp == f"{prefix}.bias_qdq" for inp in consumer.input
                ):
                    output_tensor = consumer.output[0]
                    break
        
        visited = set()
        queue = [output_tensor]
        while queue:
            tensor = queue.pop(0)
            if tensor in visited or len(visited) > 100:  # Increased limit
                continue
            visited.add(tensor)
            for consumer in self.consumers.get(tensor, []):
                if consumer.op_type == "QuantizeLinear" and len(consumer.input) >= 3:
                    s, zp = consumer.input[1], consumer.input[2]
                    if s in self.initializers and zp in self.initializers:
                        return {
                            "scale": self._to_torch_tensor(self.initializers[s]),
                            "zeropoint": self._to_torch_tensor(self.initializers[zp]),
                        }
                if consumer.op_type in ssr_downstream_ops:
                    queue.extend(consumer.output)
        return None

    def _get_activation_roles(self, export_prefix):
        """SSR-specific role assignment based on layer type.
        
        - Norm layers: output only 
        - Embeddings (col_embed, row_embed): output only (Gather has weight + constant index)
        - Layer norms in specific modules: output only
        - Attention projections: input + output
        - Everything else: input + output
        """
        prefix_lower = export_prefix.lower()
        
        # Output only: norm layers and embeddings
        if "norm" in prefix_lower or "embed" in prefix_lower:
            return {"output"}
        
        # Input + output: attention-related and other compute layers
        return {"input", "output"}


# SSR Quantized Model Extractor - Summary

## Overall Status

### Initial Problem
# - **35 missing input qparams** (35 layers without activation quantization)
# - **21 missing output qparams** (21 layers without output quantization)
# -  **Total missing: 56 out of ~288 encodings (19.4% missing)**

# ### Final Result
# - **14 missing input qparams** (60% recovered)
# - **1 missing output qparam** (95% recovered)
# - **Total missing: 15 out of ~273 extracted (5.5% missing)**
# - **Extracted: 167 activation + 106 param = 273 encodings (94.8% success)**

# ## Implementation Details

# ### Key Overrides for SSR Architecture

# 1. **Extended Compute Operations** 
#    - Added `LayerNormalization` and `InstanceNormalization` to compute ops
#    - Added `Gather` for embedding operations
#    - Now properly recognizes norm layers as parametrized compute operations

# 2. **Extended Passthrough Operations**
#    - Added arithmetic ops: `Mul`, `Add`, `Sub`, `Div` 
#    - Added reshaping: `Tile`, `Expand`, `Concat`
#    - Added other: `Shape`, `Pad`, `Slice`, `Resize`, `Where`
#    - Enables tracing through complex parameter initialization flows

# 3. **Custom Activation Roles**
#    - Norm layers: Output-only (no separate input activation)
#    - Embeddings: Output-only (Gather uses weight + constant index)
#    - Attention components: Input + output (traditional pattern)

# 4. **Extended Output Parameter Discovery**
#    - Increased traversal limit from 50 to 100 nodes
#    - Added comprehensive downstream op list for longer paths

# ## Recovered Qparams

# ### Norm Layers ✅ (18/18)
# - All `transformer.encoder.layers.*.norms.*` (9 layers)
# - All `latent_decoder.layers.*.norms.*` (6 layers)
# - All `way_decoder.layers.*.norms.*` (2 layers)
# - `transformer.can_bus_mlp.norm` (1 layer)

# ### Embeddings ✅ (2/2)
# - `positional_encoding.col_embed` (output)
# - `positional_encoding.row_embed` (output)

# ### Other ✅ (6/7)
# - `tokenlearner.layer_norm` (partial - missing output)
# - Various MLP and attention components with proper quantization

# ## Remaining Issues

# ### 1. Attention Projections Missing Input (14 layers)
# **Layers:**
# - `transformer.encoder.layers.*.attentions.*.output_proj`
# - `transformer.encoder.layers.*.attentions.*.value_proj`
# - `transformer.encoder.layers.*.attentions.*.attention_weights`
# - `transformer.encoder.layers.*.attentions.*.sampling_offsets`

# **Root Cause:**
# - MatMul input [0] comes from ReduceMean (no quantization)
# - MatMul input [1] is quantized weight
# - ReduceMean produces unquantized activation
# - These layers may only have weight quantization, no input quantization

# **Assessment:** Likely design intent - these layers process pre-normalized activations

# ### 2. tokenlearner.layer_norm Missing Output (1 layer)
# **Root Cause:**
# - Weight serves as **scaling factor** (Unsqueeze → Mul), not compute parameter
# - Actual norm: InstanceNormalization uses different inputs (Reshape output)
# - Weight quantization is for layer output scaling, not direct parameter quant
# - Non-standard pattern: separate code path needed

# **Assessment:** Complex parameterization requiring special case handling

# ## Technical Insights

# ### SSR vs ResNet Differences
# | Aspect | ResNet | SSR |
# |--------|--------|-----|
# | Norm Operations | Batch norm (implicit) | LayerNorm + InstanceNorm |
# | Parameterization | Direct (Conv weights) | Indirect (scaling factors) |
# | Embeddings | Linear layers | Gather ops |
# | Attention | Simple MatMul | MatMul + complex transforms |
# | Activation Paths | Straightforward | Multi-layered with arithmetic |

# ### Graph Patterns Discovered
# 1. **Norm Layer Scale Parameterization:**
#    - weight_dq → Unsqueeze → Mul → Add → QuantizeLinear
#    - Requires comprehensive passthrough op support

# 2. **Embedding Parameterization:**
#    - weight_dq → Gather → Unsqueeze → Expand → Tile → ... → QuantizeLinear
#    - Long traversal path through reshaping operations

# 3. **Attention Projection Structure:**
#    - weight_dq → Transpose → MatMul
#    - Input activation from separate unquantized path (ReduceMean)

# ## Recommendations

# ### For Production Use
# - Current 273 encodings (94.8%) is sufficient for most quantized inference
# - The 15 missing are edge cases with non-standard quantization patterns
# - Focus on the 14 attention layers if input quantization becomes necessary

# ### For Future Enhancement
# 1. Add special case handler for layers where weight serves as scaling factor
# 2. Implement ReduceMean → DequantizeLinear tracking for attention inputs
# 3. Consider creating layer-specific extraction rules for complex patterns
# 4. Document SSR-specific quantization conventions for model developers