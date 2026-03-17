import os, sys
import subprocess
import platform
import numpy as np
import cv2
import argparse
import audio
import shutil
import librosa
from os import listdir, path
from tqdm import tqdm
from PIL import Image
from scipy.io.wavfile import write
import gc

import onnxruntime
onnxruntime.set_default_logger_severity(3)

# face detection and alignment
from utils.retinaface import RetinaFace
from utils.face_alignment import get_cropped_head_256
detector = RetinaFace("utils/scrfd_2.5g_bnkps.onnx", provider=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"], session_options=None)

# specific face selector
from faceID.faceID import FaceRecognition
recognition = FaceRecognition('faceID/recognition.onnx')


# arguments
parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, help='Name of saved checkpoint to load weights from', required=True)
parser.add_argument('--face', type=str, help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str, help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--denoise', default=False, action="store_true", help="Denoise input audio to avoid unwanted lipmovement")
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', default='results/result_voice.mp4')
parser.add_argument('--hq_output', default=False, action='store_true',help='HQ output')

parser.add_argument('--static', default=False, action='store_true', help='If True, then use only first video frame for inference')
parser.add_argument('--pingpong', default=False, action='store_true',help='pingpong loop if audio is longer than video')

parser.add_argument('--cut_in', type=int, default=0, help="Frame to start inference")
parser.add_argument('--cut_out', type=int, default=0, help="Frame to end inference")
parser.add_argument('--fade', action="store_true", help="Fade in/out")

parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', default=25., required=False)
parser.add_argument('--resize_factor', default=1, type=int, help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--enhancer', default='none', choices=['none', 'gpen', 'gfpgan', 'codeformer', 'restoreformer'])
parser.add_argument('--blending', default=10, type=float, help='Amount of face enhancement blending 1 - 10')
parser.add_argument('--sharpen', default=False, action="store_true", help="Slightly sharpen swapped face")
parser.add_argument('--frame_enhancer', action="store_true", help="Use frame enhancer")

parser.add_argument('--face_mask', action="store_true", help="Use face mask")
parser.add_argument('--face_occluder', action="store_true", help="Use x-seg occluder face mask")

parser.add_argument('--pads', type=int, default=4, help='Padding top, bottom to adjust best mouth position, move crop up/down, between -15 to 15') # pos value mov synced mouth up
parser.add_argument('--face_mode', type=int, default=0, help='Face crop mode, 0 or 1, rect or square, affects mouth opening' )

parser.add_argument('--preview', default=False, action='store_true', help='Preview during inference')

# removed arguments
#parser.add_argument('--face_det_batch_size', type=int, help='Batch size for face detection', default=16)
#parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=1)
#parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')
#parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.''Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')
#parser.add_argument('--rotate', default=False, action='store_true',help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.''Use if you get a flipped result, despite feeding a normal looking video')
#parser.add_argument('--nosmooth', default=False, action='store_true',help='Prevent smoothing face detections over a short temporal window')

args = parser.parse_args()

if args.checkpoint_path == 'checkpoints\wav2lip_384.onnx' or args.checkpoint_path == 'checkpoints\wav2lip_384_fp16.onnx':
	args.img_size = 384
else:
	args.img_size = 96

mel_step_size = 16
padY = max(-15, min(args.pads, 15))

device = 'cpu'
if onnxruntime.get_device() == 'GPU':
		device = 'cuda'
print("Running on " + device)


if args.enhancer == 'gpen':
		from enhancers.GPEN.GPEN import GPEN
		enhancer = GPEN(model_path="enhancers/GPEN/GPEN-BFR-256-sim.onnx", device=device) #GPEN-BFR-256-sim

if args.enhancer == 'codeformer':
		from enhancers.Codeformer.Codeformer import CodeFormer
		enhancer = CodeFormer(model_path="enhancers/Codeformer/codeformerfixed.onnx", device=device)
    
if args.enhancer == 'restoreformer':
		from enhancers.restoreformer.restoreformer16 import RestoreFormer
		enhancer = RestoreFormer(model_path="enhancers/restoreformer/restoreformer16.onnx", device=device)
		
if args.enhancer == 'gfpgan':
		from enhancers.GFPGAN.GFPGAN import GFPGAN
		enhancer = GFPGAN(model_path="enhancers/GFPGAN/GFPGANv1.4.onnx", device=device)

if args.frame_enhancer:
		from enhancers.RealEsrgan.esrganONNX import RealESRGAN_ONNX
		frame_enhancer = RealESRGAN_ONNX(model_path="enhancers/RealEsrgan/clear_reality_x4.onnx", device=device)
    
if args.face_mask:
		from blendmasker.blendmask import BLENDMASK
		masker = BLENDMASK(model_path="blendmasker/blendmasker.onnx", device=device)
		
if args.face_occluder:
		from xseg.xseg import MASK
		occluder = MASK(model_path="xseg/xseg.onnx", device=device)

if args.denoise:
	from resemble_denoiser.resemble_denoiser import ResembleDenoiser
	denoiser = ResembleDenoiser(model_path='resemble_denoiser/denoiser.onnx', device=device)
				        		    
if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
		args.static: args.static = True



def load_model(device):
	model_path = args.checkpoint_path
	session_options = onnxruntime.SessionOptions()
	session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
	providers = ["CPUExecutionProvider"]
	if device == 'cuda':
		providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),"CPUExecutionProvider"]
		
	session = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers)	
	
	return session


def select_specific_face(model, spec_img, size, crop_scale=1.0):

    # select face:
		h, w = spec_img.shape[:-1]
		roi = cv2.selectROI("Select speaker face", spec_img, showCrosshair=False)
		if roi == (0,0,0,0):roi = (0,0,w,h)
		cropped_roi = spec_img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
		cv2.destroyAllWindows()
		
		bboxes, kpss = model.detect(cropped_roi, input_size = (320,320), det_thresh=0.3)
		assert len(kpss) != 0, "No face detected"
		
		target_face, mat = get_cropped_head_256(cropped_roi, kpss[0], size=size, scale=crop_scale)
		target_face = cv2.resize(target_face,(112,112))
		target_id = recognition(target_face)[0].flatten()

		return target_id
    
def process_video_specific(model, img, size, target_id, crop_scale=1.0):
		ori_img = img
		bboxes, kpss = model.detect(ori_img, input_size=(320, 320), det_thresh=0.3)
    
		assert len(kpss) != 0, "No face detected"

		best_score = -float('inf')
		best_aimg = None
		best_mat = None

		for kps in kpss:
				aimg, mat = get_cropped_head_256(ori_img, kps, size=size, scale=crop_scale)
        
				face = aimg.copy()
				face = cv2.resize(face, (112, 112))
				face_id = recognition(face)[0].flatten()
        
        # Calculate similarity score with the target ID
				score = target_id @ face_id  # Dot product or cosine similarity
                
				if score > best_score:
						best_score = score
						best_aimg = aimg
						best_mat = mat
				if best_score < 0.4:
						best_aimg = np.zeros((256,256), dtype=np.uint8)
						best_aimg = cv2.cvtColor(best_aimg, cv2.COLOR_GRAY2RGB)/255
						best_mat = np.float32([[1,2,3],[1,2,3]])                          

		return best_aimg, best_mat
        
def face_detect(images, target_id):

	os.system('cls')
	print ("Detecting face and generating data...")
					
	crop_size = 256

	sub_faces = []
	crop_faces = []
	matrix = []
	face_error = []
				
	for i in tqdm(range(0, len(images))):

		try:

			crop_face, M = process_video_specific(detector, images[i], 256, target_id, crop_scale=1.0)

      # crop modes
			if args.face_mode == 0:
				sub_face = crop_face[65-(padY):241-(padY),62:194]
				#cv2.imwrite("sub_0.jpg",sub_face)
			else:
				sub_face = crop_face[65-(padY):241-(padY),42:214]
				#cv2.imwrite("sub_1.jpg",sub_face)
			
			sub_face = cv2.resize(sub_face, (args.img_size,args.img_size))
  
			sub_faces.append(sub_face)		
			crop_faces.append(crop_face)
			matrix.append(M)
  
			no_face = 0
  		
		except:
			if i == 0:
				crop_face = np.zeros((256,256), dtype=np.uint8)
				crop_face = cv2.cvtColor(crop_face, cv2.COLOR_GRAY2RGB)/255
				sub_face = crop_face[65-(padY):241-(padY),62:194]
				sub_face = cv2.resize(sub_face, (args.img_size,args.img_size))
				M = np.float32([[1,2,3],[1,2,3]])
  							
			sub_faces.append(sub_face)		
			crop_faces.append(crop_face)
			matrix.append(M)
  		
			no_face = -1
			
		face_error.append(no_face)
		
	return crop_faces, sub_faces, matrix, face_error 

def datagen(frames, mels):
	
	img_batch, mel_batch, frame_batch = [], [], []

	for i, m in enumerate(mels):

		idx = 0 if args.static else i%len(frames)

		frame_to_save = frames[idx].copy()
		frame_batch.append(frame_to_save)
			
		img_batch.append(frames[idx])
		mel_batch.append(m)

		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0
		#img_masked[:, :, args.img_size // 2:, :] = 0
		
		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch
		img_batch, mel_batch, frame_batch = [], [], []

def main():
	if args.hq_output:
		if not os.path.exists('hq_temp'):
			os.mkdir('hq_temp')

  # ffmpeg preset for HQ processing	
	preset='medium'
	 
	blend = args.blending/10
 
	static_face_mask = np.zeros((224,224), dtype=np.uint8)
	static_face_mask = cv2.ellipse(static_face_mask, (112,162), (62,54),0,0,360,(255,255,255), -1)
	static_face_mask = cv2.ellipse(static_face_mask, (112,122), (46,23),0,0,360,(0,0,0), -1)
	static_face_mask = cv2.resize(static_face_mask,(256,256))
	
	static_face_mask = cv2.rectangle(static_face_mask, (0,246), (246,246),(0,0,0), -1)
	static_face_mask = cv2.cvtColor(static_face_mask, cv2.COLOR_GRAY2RGB)/255
	static_face_mask = cv2.GaussianBlur(static_face_mask,(19,19),cv2.BORDER_DEFAULT)

	sub_face_mask = np.zeros((256,256), dtype=np.uint8)
	
	#if args.face_mode == 0:
	#	sub_face_mask = cv2.rectangle(sub_face_mask, (62, 65 - padY), (194, 241 - padY), (255, 255, 255), -1) #0
	#else:
	#	sub_face_mask = cv2.rectangle(sub_face_mask, (42, 65 - padY), (214, 241 - padY), (255, 255, 255), -1) #1
	
	sub_face_mask = cv2.rectangle(sub_face_mask, (42, 65 - padY), (214, 249), (255, 255, 255), -1) #1
	sub_face_mask = cv2.GaussianBlur(sub_face_mask.astype(np.uint8),(29,29),cv2.BORDER_DEFAULT)
	sub_face_mask = cv2.cvtColor(sub_face_mask, cv2.COLOR_GRAY2RGB)		
	sub_face_mask = sub_face_mask/255
		
	im = cv2.imread(args.face)

	if not os.path.isfile(args.face):
		raise ValueError('--face argument must be a valid path to video/image file')
	
	elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg', 'bmp']:
		orig_frame = cv2.imread(args.face)
		orig_frame = cv2.resize(orig_frame, (orig_frame.shape[1]//args.resize_factor, orig_frame.shape[0]//args.resize_factor))	
		orig_frames = [orig_frame]
		fps = args.fps

    # crop final:
		h, w = orig_frame.shape[:-1]
		roi = cv2.selectROI("Crop final video", orig_frame, showCrosshair=False)
		if roi == (0,0,0,0):roi = (0,0,w,h)
		cropped_roi = orig_frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
		cv2.destroyAllWindows()
		full_frames = [cropped_roi]
		orig_h, orig_w = cropped_roi.shape[:-1]
		
    #	select specific face:
		target_id = select_specific_face(detector, cropped_roi, 256, crop_scale=1)
								
	else:
		video_stream = cv2.VideoCapture(args.face)
		fps = video_stream.get(cv2.CAP_PROP_FPS)
		video_stream.set(1,args.cut_in)

		print('Reading video frames...')
		
    # cut to input/putput position:
		if args.cut_out == 0:
			args.cut_out = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
			
		duration = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT)) - args.cut_in
		new_duration = args.cut_out - args.cut_in
		
		if args.static:
			new_duration = 1
	
		video_stream.set(1,args.cut_in)
		
    # read frames and crop roi:
		full_frames = []
		orig_frames = []
		
		for l in range(new_duration):
			still_reading, frame = video_stream.read()
			
			if not still_reading:
				video_stream.release()
				break
				
			if args.resize_factor > 1:
				frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))			

      # crop first frame:
			if l == 0:
				h, w = frame.shape[:-1]
				roi = cv2.selectROI("Crop final video", frame, showCrosshair=False)
				if roi == (0,0,0,0):roi = (0,0,w,h)
								
				cropped_roi = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
				cv2.destroyAllWindows()
				os.system('cls')
				
        # select_specific_face:
				target_id = select_specific_face(detector, cropped_roi, 256, crop_scale=1)
				orig_h, orig_w = cropped_roi.shape[:-1]
				print("Reading frames....")
			print(f'\r{l}', end=' ', flush=True)
			
      # crop all frames:
			cropped_roi = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
			full_frames.append(cropped_roi)
			orig_frames.append(cropped_roi)			

  # memory usage raw video:
	memory_usage_bytes = sum(frame.nbytes for frame in full_frames)
	memory_usage_mb = memory_usage_bytes / (1024**2)
	
	print ("Number of frames used for inference: " + str(len(full_frames)) + " / ~ " + str(int(memory_usage_mb)) + " mb memory usage")
	
  
  # convert input audio to wav anyway:
	print('Extracting raw audio...')
	subprocess.run(['ffmpeg', '-y', '-i', args.audio, '-ac', '1', '-strict', '-2', 'temp/temp.wav'])

	os.system('cls')
	print('Raw audio extracted')

  # denoise extracted audio:
	if args.denoise:
		print('Denoising audio...')
		wav, sr = librosa.load('temp/temp.wav', sr=44100, mono=True)
		wav_denoised, new_sr = denoiser.denoise(wav, sr, batch_process_chunks=False)
		write('temp/temp.wav', new_sr, (wav_denoised * 32767).astype(np.int16))
		try:
			if hasattr(denoiser, 'session'):
				del denoiser.session
				gc.collect()
		except:
			pass

	wav = audio.load_wav('temp/temp.wav', 16000)
	mel = audio.melspectrogram(wav)

	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

	mel_chunks = []
	mel_idx_multiplier = 80./fps 
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1

	print("Length of mel chunks: {}".format(len(mel_chunks)))

	full_frames = full_frames[:len(mel_chunks)]

  # face detection:	
	aligned_faces, sub_faces, matrix, no_face = face_detect(full_frames, target_id)

	if args.pingpong:
		orig_frames = orig_frames + orig_frames[::-1]
		full_frames = full_frames + full_frames[::-1]
		aligned_faces = aligned_faces + aligned_faces[::-1]
		sub_faces = sub_faces + sub_faces[::-1]
		matrix = matrix + matrix[::-1]
		no_face = no_face + no_face[::-1]

  # datagen:					
	gen = datagen(sub_faces.copy(), mel_chunks)
	
	fc = 0

	model = load_model(device)

	frame_h, frame_w = full_frames[0].shape[:-1]

	out = cv2.VideoWriter('temp/temp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (orig_w, orig_h))
				
	os.system('cls')
	print('Running on ' + onnxruntime.get_device())
	print ('Checkpoint: ' + args.checkpoint_path)
	print ('Resize factor: ' + str(args.resize_factor))
	if args.pingpong: print ('Use pingpong')
	if args.enhancer != 'none': print ('Use ' + args.enhancer)
	if args.face_mask: print ('Use face mask')
	if args.face_occluder: print ('Use occlusion mask')
	print ('')

  # fade in/out
	fade_in = 11
	total_length = int(np.ceil(float(len(mel_chunks))))
	fade_out = total_length - 11
	bright_in = 0
	bright_out = 0
	
	for i, (img_batch, mel_batch, frames) in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks)))))):
					
		if fc == (len(full_frames)):
			fc = 0
		
		face_err = no_face[fc]
		
		img_batch = img_batch.transpose((0, 3, 1, 2)).astype(np.float32)
		mel_batch = mel_batch.transpose((0, 3, 1, 2)).astype(np.float32)
		
    # wav2lip onnx inference:
		pred = model.run(None,{'mel_spectrogram':mel_batch, 'video_frames':img_batch})[0][0]
				
		pred = pred.transpose(1, 2, 0)*255
		pred = pred.astype(np.uint8)
		pred = pred.reshape((1, args.img_size, args.img_size, 3))		
		
		mat = matrix[fc]
		mat_rev = cv2.invertAffineTransform(mat)

		aligned_face = aligned_faces[fc]
		aligned_face_orig = aligned_face.copy()
		p_aligned = aligned_face.copy()
		
		full_frame = full_frames[fc]

		final = orig_frames[fc]
	
		for p, f in zip(pred, frames):			
				
			if not args.static: fc = fc + 1

      # crop mode:
			if args.face_mode == 0:
				p = cv2.resize(p,(132,176))
			else:
				p = cv2.resize(p,(172,176))

				
			if args.face_mode == 0:
				p_aligned[65-(padY):241-(padY),62:194] = p
			else:
				p_aligned[65-(padY):241-(padY),42:214] = p
			
			aligned_face = (sub_face_mask * p_aligned + (1 - sub_face_mask) * aligned_face_orig).astype(np.uint8)
			
			if face_err != 0:
				res = full_frame
				face_err = 0
				
			else:
			
        # face enhancers:
				if args.enhancer != 'none':      
					aligned_face_enhanced = enhancer.enhance(aligned_face)
					aligned_face_enhanced = cv2.resize(aligned_face_enhanced,(256,256))
					aligned_face = cv2.addWeighted(aligned_face_enhanced.astype(np.float32),blend, aligned_face.astype(np.float32), 1.-blend, 0.0)        
        					
        # mask options:
				if args.face_mask:
					seg_mask = masker.mask(aligned_face)
					#seg_mask[seg_mask > 32] = 255
					seg_mask = cv2.blur(seg_mask,(5,5))					
					seg_mask = seg_mask /255
					mask = cv2.warpAffine(seg_mask, mat_rev,(frame_w, frame_h))
					
				if args.face_occluder:
          # handle specific face not detected:
					try:
						seg_mask = occluder.mask(aligned_face_orig)
						seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2RGB)					
						mask = cv2.warpAffine(seg_mask, mat_rev,(frame_w, frame_h))
					except:
						seg_mask = occluder.mask(aligned_face) #xseg
						seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2RGB)					
						mask = cv2.warpAffine(seg_mask, mat_rev,(frame_w, frame_h))
				  
				if not args.face_mask and not args.face_occluder:
					mask = cv2.warpAffine(static_face_mask, mat_rev,(frame_w, frame_h))		
	
				if args.sharpen:
					#smoothed = cv2.GaussianBlur(aligned_face, (9, 9), 10)
					#aligned_face = cv2.addWeighted(aligned_face, 1.5, smoothed, -0.5, 0)
					#aligned_face = np.clip(aligned_face, 0, 255).astype(np.uint8)
					aligned_face = cv2.detailEnhance(aligned_face, sigma_s=1.3, sigma_r=0.15)
				
				#cv2.imshow("D",aligned_face)
				
				dealigned_face =  cv2.warpAffine(aligned_face, mat_rev, (frame_w, frame_h))
				#cv2.imshow("mask",mask)
				#cv2.waitKey(1)
				#mask = cv2.warpAffine(static_face_mask, mat_rev,(frame_w, frame_h))
				
				res = (mask * dealigned_face + (1 - mask) * full_frame).astype(np.uint8)

		final = res

		if args.frame_enhancer:
			final = frame_enhancer.enhance(final)
			final = cv2.resize(final,(orig_w, orig_h), interpolation=cv2.INTER_AREA)
            
    # fade in/out:
		if i < 11 and args.fade:
			final = cv2.convertScaleAbs(final, alpha=0 + (0.1 * bright_in), beta=0)
			bright_in = bright_in + 1		
		if i > fade_out and args.fade:
			final = cv2.convertScaleAbs(final, alpha=1 - (0.1 * bright_out), beta=0)
			bright_out = bright_out + 1
					
		if args.hq_output:
			cv2.imwrite(os.path.join('./hq_temp', '{:0>7d}.png'.format(i)), final)
		else:	
			out.write(final)

		if args.preview:
			cv2.imshow("Result - press ESC to stop and save",final)
			k = cv2.waitKey(1)
			if k == 27:
				cv2.destroyAllWindows()
				out.release()
				break

			if k == ord('s'):
				if args.sharpen == False:
					args.sharpen = True
				else:
					args.sharpen = False
				print ('')    
				print ("Sharpen = " + str(args.sharpen))
						
	out.release()

	if args.hq_output:
		 command = 'ffmpeg.exe -y -i ' + '"' + args.audio + '"' + ' -r ' + str(fps) + ' -f image2 -i ' + '"' + './hq_temp/' + '%07d.png' + '"' + ' -shortest -vcodec libx264 -pix_fmt yuv420p -crf 5 -preset slow -acodec libmp3lame -ac 2 -ar 44100 -ab 128000 -strict -2 ' + '"' + args.outfile + '"'
	else:						
		command = 'ffmpeg.exe -y -i ' + '"' + args.audio + '"' + ' -i ' + 'temp/temp.mp4' + ' -shortest -vcodec copy -acodec libmp3lame -ac 2 -ar 44100 -ab 128000 -strict -2 ' + '"' + args.outfile + '"'

	subprocess.call(command, shell=platform.system() != 'Windows')
		
	if os.path.exists('temp/temp.mp4'):
		os.remove('temp/temp.mp4')
	if os.path.exists('temp/temp.wav'):
		os.remove('temp/temp.wav')		
	if  os.path.exists('hq_temp'):
		shutil.rmtree('hq_temp')	

if __name__ == '__main__':
	main()
