import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download


# 1. 모델 로드
def load_model_from_huggingface(repo_id, filename):
    try:
        # 모델 파일을 로컬에 다운로드
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"모델이 {model_path}에 다운로드되었습니다.")
        # Keras 모델 로드
        model = tf.keras.models.load_model(model_path)
        print("모델이 성공적으로 로드되었습니다.")
        return model
    except Exception as e:
        print(f"Hugging Face에서 모델을 로드하는 중 오류가 발생했습니다: {e}")
        raise e

# 이미지 전처리
def preprocess_image(image):
    # image = image.convert('RGB')
    # image = image.resize((128, 128))
    # array = tf.keras.preprocessing.image.img_to_array(image)
    array = np.array(image)
    array = np.expand_dims(array, axis=0) / 255.0  # 정규화
    
    return array