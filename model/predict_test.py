import bentoml
import numpy as np
import torch

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    tensor = torch.rand(128, 128, 4) * 255
    tensor_float16 = tensor.to(torch.float16)
    result = client.generate_frames(
        input_array=tensor_float16
    )
    print(type(result.shape))
    print("Array shape:", result.shape)
    print(result[0,80,:,:])