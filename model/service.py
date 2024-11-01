import bentoml
import os
from bentoml.io import Image, JSON
from transformers import AutoTokenizer
import tensorflow as tf
import numpy as np

from loss import mse_rgb
from preprocessing import preprocess_image

# 1. 허깅페이스 허브에서 Keras 기반 모델 로드
model_name = 'unetv2_rgbmse.keras'
model_path = os.path.join(os.path.dirname(__file__), model_name)  # 사용하려는 모델 이름으로 변경
model = tf.keras.models.load_model(model_path)

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
        generated_frames = self.artifacts.model.predict(input_array)  # (10, 128, 128, 4)
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
