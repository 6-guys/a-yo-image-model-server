import numpy as np

# 이미지 전처리
def preprocess_image(image):
    # image = image.convert('RGB')
    # image = image.resize((128, 128))
    # array = tf.keras.preprocessing.image.img_to_array(image)
    array = np.array(image)
    array = np.expand_dims(array, axis=0) / 255.0  # 정규화
    
    return array