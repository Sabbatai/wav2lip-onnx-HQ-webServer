import numpy as np
import onnxruntime
from librosa import stft, istft

class ResembleDenoiser:
    def __init__(self, model_path='denoiser_fp16.onnx', device='cpu'):
        self.stft_hop_length = 420
        self.win_length = self.n_fft = 4 * self.stft_hop_length

        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.inter_op_num_threads = 4
        session_options.intra_op_num_threads = 4
        session_options.log_severity_level = 4

        providers = ["CPUExecutionProvider"]
        if device == 'cuda':
            providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]

        self.session = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers)


    def _stft(self, x):
        s = stft(
            x, window='hann', win_length=self.win_length, n_fft=self.n_fft,
            hop_length=self.stft_hop_length, center=True, pad_mode='reflect'
        )
        s = s[..., :-1]
        mag = np.abs(s)
        phi = np.angle(s)
        return mag, np.cos(phi), np.sin(phi)

    def _istft(self, mag, cos, sin):
        real = mag * cos
        imag = mag * sin
        s = real + imag * 1.0j
        s = np.pad(s, ((0, 0), (0, 0), (0, 1)), mode='edge')
        x = istft(
            s, window='hann', win_length=self.win_length,
            hop_length=self.stft_hop_length, n_fft=self.n_fft
        )
        return x

    def _model_infer(self, wav):
        padded_wav = np.pad(wav, ((0, 0), (0, 441)))
        mag, cos, sin = self._stft(padded_wav)

        ort_inputs = {
            "mag": mag,
            "cos": cos,
            "sin": sin,
        }
        sep_mag, sep_cos, sep_sin = self.session.run(None, ort_inputs)
        out = self._istft(sep_mag, sep_cos, sep_sin)
        return out[:wav.shape[-1]]

    def denoise(self, wav: np.ndarray, sample_rate: int, batch_process_chunks=False):
        assert wav.ndim == 1, 'Input should be 1D (mono) wav'

        chunk_length = int(44100 * 30)
        hop_length = chunk_length
        num_chunks = 1 + (wav.shape[-1] - 1) // hop_length
        n_pad = (num_chunks - wav.shape[-1] % num_chunks) % num_chunks
        wav = np.pad(wav, (0, n_pad))

        chunks = np.reshape(wav, (num_chunks, -1))
        abs_max = np.clip(np.max(np.abs(chunks), axis=-1, keepdims=True), 1e-7, None)
        chunks /= abs_max

        if batch_process_chunks:
            res_chunks = self._model_infer(chunks)
        else:
            res_chunks = np.array([
                self._model_infer(c[None]) for c in chunks
            ]).squeeze(axis=1)

        res_chunks *= abs_max
        res = np.reshape(res_chunks, (-1))
        return res[:wav.shape[-1]], 44100
