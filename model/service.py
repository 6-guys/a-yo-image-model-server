import bentoml
import numpy as np
import torch
from bentoml.validators import Shape, DType
from pydantic import Field
from typing import Annotated  # Python 3.9 or above
from loss import mse_rgb
from preprocessing import preprocess_image, load_model_from_huggingface


# BentoML service definition
@bentoml.service(
    resources={"cpu": "4"},
    traffic={"timeout": 180},
)

class FrameGenerationService:
    
    def __init__(self) -> None:
        # Model download and load
        repo_id = "mk48/nipa-cunet"
        model_filename = "unetv2_rgbmse.keras"
        self.model = load_model_from_huggingface(repo_id, model_filename)

    @bentoml.api
    def generate_frames(self,         
            input_array: Annotated[torch.Tensor, Shape((128, 128, 4)), DType("float16")]
            = Field(description="A 128x128x4 tensor with float16 dtype")) -> np.ndarray :
        
        # Model prediction
        generated_frames = self.model.predict(input_array)  # (128, 128, 4)
        generated_frames = (generated_frames * 255).astype(np.float16).squeeze().tolist()  # 리스트로 변환
        
        return {"generated_frames": generated_frames}
