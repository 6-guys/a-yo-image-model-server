import bentoml
import os
from bentoml.io import Image, JSON
from transformers import AutoTokenizer
from PIL import Image as PILImage
import tensorflow as tf
import numpy as np

from loss import mse_rgb
from preprocessing import preprocess_image, load_model_from_huggingface

# 모델 다운로드 및 로드
repo_id = "mk48/nipa-cunet"
model_filename = "unetv2_rgbmse.keras"
model = load_model_from_huggingface(repo_id, model_filename)


# 2. BentoML 서비스 정의
@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([
    bentoml.artifact.TensorflowModelArtifact('model'),
])
class FrameGenerationService(bentoml.BentoService):
    
    @bentoml.api(input=Image(), output=JSON())
    def generate_frames(self, input_image):
        input_array = preprocess_image(input_image)
        
        # 모델 예측
        generated_frames = self.artifacts.model.predict(input_array)  # (10, 128, 128, 3)
        generated_frames = (generated_frames * 255).astype(np.uint8).squeeze().tolist()  # 리스트로 변환
        
        return {"generated_frames": generated_frames}

# 3. 서비스 인스턴스 생성 및 아티팩트 저장
if __name__ == "__main__":
    # 서비스 인스턴스 생성
    service = FrameGenerationService()
    
    # 모델 및 아티팩트 저장
    service.pack('model', model)
    # service.pack('tokenizer', tokenizer)
    
    # 서비스 저장
    saved_path = service.save()
    print(f"Service saved to: {saved_path}")
