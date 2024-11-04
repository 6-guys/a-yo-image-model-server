import bentoml
import numpy as np
import torch

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    test_random_array = np.random.rand(128, 128, 4) * 255
    result = client.generate_frames(
        input_array=test_random_array
    )
    print(type(result.shape))
    print("Array shape:", result.shape)
    print(result[0,80,:,:])