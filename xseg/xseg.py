import cv2
import numpy as np
import onnxruntime

class MASK:
    def __init__(self, model_path="xseg.onnx", device='cpu'):
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        if device == 'cuda':
            providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),"CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers)

        
    def mask(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = img.astype(np.float32)
        img = img / 255
        img = np.expand_dims(img, axis=0).astype(np.float32) 
              
        result = self.session.run(None, {(self.session.get_inputs()[0].name):img})[0][0]

        return result
