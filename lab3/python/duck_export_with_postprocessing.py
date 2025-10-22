import torch
import torch.nn as nn
from transformers import Mask2FormerForUniversalSegmentation


class Mask2FormerWithPostProcessing(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        outputs = self.model(pixel_values)
        
        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits

        # Scale back to preprocessed image size - (384, 384) for all models
        masks_queries_logits = nn.functional.interpolate(
            masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
        )

        # Apply softmax to class logits
        class_scores = nn.functional.softmax(class_queries_logits, dim=-1)

        # Apply sigmoid to mask logits
        mask_probs = masks_queries_logits.sigmoid()

        return class_scores, mask_probs


# Load model
model = Mask2FormerForUniversalSegmentation.from_pretrained("./model")
model.eval()

# Create the new model with post-processing
model_with_postprocessing = Mask2FormerWithPostProcessing(model)
model_with_postprocessing.eval()


# Export model
dummy_input = torch.randn(1, 3, 512, 512)

# Export as TorchScript model
with torch.no_grad():
    traced_model = torch.jit.trace(model_with_postprocessing, dummy_input)
    torch.jit.save(traced_model, "../data/duck_with_postprocessing.pt")

# There are some issues with tracing HF models on one device (e.g., CPU) and running inference on another device (e.g., MPS, CUDA).
#
# See also:
# - https://github.com/huggingface/transformers/issues/25261
# - https://github.com/huggingface/transformers/issues/5664
# - https://github.com/huggingface/transformers/issues/22038
# - https://github.com/pytorch/pytorch/issues/50971


# Export as ONNX model
torch.onnx.export(model_with_postprocessing, 
                  dummy_input, 
                  "../data/duck_with_postprocessing.onnx", 
                  opset_version=16, 
                  input_names=['input'], 
                  output_names=['class_scores', 'mask_probs'],
                  dynamic_axes={
                      'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                      'class_scores': {0: 'batch_size'},
                      'mask_probs': {0: 'batch_size'}
                  },
                  do_constant_folding=True,
                  export_params=True,
                  keep_initializers_as_inputs=None,
                  verbose=True)

