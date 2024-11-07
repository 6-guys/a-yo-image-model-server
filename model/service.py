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
            input_array: Annotated[np.ndarray, Shape((128, 128, 4)), DType("float16")]
            = Field(description="A 128x128x4 tensor with float16 dtype")) -> np.ndarray :
        label = np.zeros((9 ,10))
        for i in range(9) :
            label[i, i + 1] = 1
        input_img = np.expand_dims(input_array, axis=0)
        input_img = np.repeat(input_img , 9 , axis = 0)
        # Model prediction
        generated_frames = self.model.predict([input_img , label])  # (128, 128, 4)
        print(generated_frames.shape)

        # Assuming generated_frames is your array
        generated_frames = np.clip(generated_frames, 0, 255).astype(np.uint8)

        generated_frames = generated_frames.tolist()  # 리스트로 변환
        return generated_frames
