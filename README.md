# wav2lip-onnx-HQ

Update 09.06.2025

- added garbage collection to free ram/vram after denoising audio  
  tested on RTX3060/6Gb running inference using audio denoiser, occlusion mask, face enhancer and frame enhancer at same time
  

Update 28.05.2025  

- removed 'final audio' option stuff
- added resemble audio denoiser to avoid unwanted lip movements  
  (not as good as vocal separation (eg. KimVocal_v2) but working similar in most cases)
- minor code optimizations
  

Update 29.04.2025 (inference_onnxModel_V2.py)

  - replaced occlusion mask with xseg occlusion
  - added option frame enhancer realEsrgan (clear_reality_x4 model included)
  - added option short fade-in/fade-out
  - added option for facemode 0 or 1 for better result on different face shapes  
    (0=portrait like orig. wav2lip, 1=square for less mouth opening)
  - bugfix crashing when using xseg and specific face is not detected  

Update 08.02.2025

  - optmized occlusion mask
  - Replaced insightface with retinaface detection/alignment for easier installation
  - Replaced seg-mask with faster blendmasker
  - Added free cropping of final result video
  - Added specific target face selection from first frame

.

Just another Wav2Lip HQ local installation, fully running on Torch to ONNX converted models for:
- face-detection
- face-recognition
- face-alignment
- face-parsing
- face-enhancement
- wav2lip inference.

.

Can be run on CPU or Nvidia GPU

I've made some modifications such as:
* New face-detection and face-alignment code. (working for ~ +- 60º head tilt)
* Four different face enhancers available, adjustable enhancement level .
* Choose pingpong loop instead of original loop function.
* Set cut-in/cut-out position to create the loop or cut longer video.
* Cut-in position = used frame if static is selected.
* Select the target face.
* Use two audio files, eg. vocal for driving and full music mix for final output.
* This version does not crash if no face is detected, it just continues ...

Type --help for all commandline parameters

.
 
Model download - https://drive.google.com/drive/folders/1BGl9bmMtlGEMx_wwKufJrZChFyqjnlsQ?usp=sharing  

.


Original wav2lip - https://github.com/Rudrabha/Wav2Lip

Face enhancers taken from -  https://github.com/harisreedhar/Face-Upscalers-ONNX

Face detection taken from - https://github.com/neuralchen/SimSwap

Face occluder taken from - https://github.com/facefusion/facefusion-assets/releases

Blendmasker extracted from - https://github.com/mapooon/BlendFace during onnx conversion

Face recognition for specifc face taken from - https://github.com/jahongir7174/FaceID  

Resemble-denoiser-ONNX adopted from - https://github.com/skeskinen/resemble-denoise-onnx-inference

.

.


