"""
Wav2Lip Server - Keeps models loaded in memory for fast inference
Run this server once, then send HTTP requests from chatbot
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import subprocess
import platform
import numpy as np
import cv2
from tqdm import tqdm
import onnxruntime

# Face detection and alignment
from utils.retinaface import RetinaFace
from utils.face_alignment import get_cropped_head_256
from faceID.faceID import FaceRecognition
import audio

app = FastAPI()

# Configuration
CHECKPOINT_PATH = "checkpoints/wav2lip_gan.onnx"
IMG_SIZE = 96
FPS = 15

# Global model storage - loaded once at startup
detector = None
recognition = None
wav2lip_model = None
device = 'cuda'

print("="*60)
print("🎬 INITIALIZING WAV2LIP SERVER")
print("="*60)

def initialize_models():
    """Load all models into memory once"""
    global detector, recognition, wav2lip_model
    
    print("\n⏳ Loading face detection...")
    onnxruntime.set_default_logger_severity(3)
    detector = RetinaFace(
        "utils/scrfd_2.5g_bnkps.onnx",
        provider=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"],
        session_options=None
    )
    print("✅ Face detection loaded")
    
    print("\n⏳ Loading face recognition...")
    recognition = FaceRecognition('faceID/recognition.onnx')
    print("✅ Face recognition loaded")
    
    print("\n⏳ Loading Wav2Lip model...")
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]
    if device == 'cuda':
        providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
    
    wav2lip_model = onnxruntime.InferenceSession(
        CHECKPOINT_PATH, 
        sess_options=session_options, 
        providers=providers
    )
    print("✅ Wav2Lip model loaded")
    print(f"\n🚀 Server ready! Models cached in VRAM")

@app.on_event("startup")
async def startup_event():
    """Initialize models when server starts"""
    initialize_models()

class VideoRequest(BaseModel):
    audio_path: str
    avatar_path: str
    output_path: str
    fps: int = 15

def select_target_face(detector, recognition, first_frame):
    """Auto-detect the primary face"""
    bboxes, kpss = detector.detect(first_frame, input_size=(320, 320), det_thresh=0.3)
    
    if len(kpss) == 0:
        raise ValueError("No face detected in video")
    
    target_face, mat = get_cropped_head_256(first_frame, kpss[0], size=256, scale=1.0)
    target_face = cv2.resize(target_face, (112, 112))
    target_id = recognition(target_face)[0].flatten()
    
    return target_id

def process_video_specific(model, img, size, target_id, crop_scale=1.0):
    """Process video frame and find matching face"""
    bboxes, kpss = model.detect(img, input_size=(320, 320), det_thresh=0.3)
    
    if len(kpss) == 0:
        black_face = np.zeros((256, 256, 3), dtype=np.uint8)
        mat = np.float32([[1, 0, 0], [0, 1, 0]])
        return black_face, mat
    
    best_score = -float('inf')
    best_aimg = None
    best_mat = None
    
    for kps in kpss:
        aimg, mat = get_cropped_head_256(img, kps, size=size, scale=crop_scale)
        face = cv2.resize(aimg.copy(), (112, 112))
        face_id = recognition(face)[0].flatten()
        score = target_id @ face_id
        
        if score > best_score:
            best_score = score
            best_aimg = aimg
            best_mat = mat
    
    if best_score < 0.4 or best_aimg is None:
        black_face = np.zeros((256, 256, 3), dtype=np.uint8)
        mat = np.float32([[1, 0, 0], [0, 1, 0]])
        return black_face, mat
    
    return best_aimg, best_mat

def face_detect(images, target_id, pads=4):
    """Detect faces in all frames"""
    padY = max(-15, min(pads, 15))
    
    aligned_faces = []
    sub_faces = []
    matrix = []
    face_error = []
    
    for img in tqdm(images, desc="Detecting faces"):
        try:
            crop_face, M = process_video_specific(detector, img, 256, target_id, crop_scale=1.0)
            sub_face = crop_face[65-padY:241-padY, 62:194]
            sub_face = cv2.resize(sub_face, (IMG_SIZE, IMG_SIZE))
            
            aligned_faces.append(crop_face)
            sub_faces.append(sub_face)
            matrix.append(M)
            face_error.append(0)
            
        except:
            if len(aligned_faces) > 0:
                aligned_faces.append(aligned_faces[-1])
                sub_faces.append(sub_faces[-1])
                matrix.append(matrix[-1])
            else:
                black = np.zeros((256, 256, 3), dtype=np.uint8)
                sub = cv2.resize(black[65-padY:241-padY, 62:194], (IMG_SIZE, IMG_SIZE))
                aligned_faces.append(black)
                sub_faces.append(sub)
                matrix.append(np.float32([[1, 0, 0], [0, 1, 0]]))
            face_error.append(-1)
    
    return aligned_faces, sub_faces, matrix, face_error

def datagen(frames, mels, static=False):
    """Generate batches for inference"""
    img_batch, mel_batch, frame_batch = [], [], []
    
    for i, m in enumerate(mels):
        idx = 0 if static else i % len(frames)
        
        frame_to_save = frames[idx].copy()
        frame_batch.append(frame_to_save)
        img_batch.append(frames[idx])
        mel_batch.append(m)
        
        img_batch_np = np.asarray(img_batch)
        mel_batch_np = np.asarray(mel_batch)
        
        img_masked = img_batch_np.copy()
        img_masked[:, IMG_SIZE//2:] = 0
        img_batch_final = np.concatenate((img_masked, img_batch_np), axis=3) / 255.
        mel_batch_final = np.reshape(mel_batch_np, [len(mel_batch_np), mel_batch_np.shape[1], mel_batch_np.shape[2], 1])
        
        yield img_batch_final, mel_batch_final, frame_batch
        img_batch, mel_batch, frame_batch = [], [], []

@app.post("/generate")
async def generate_video(request: VideoRequest):
    """
    Generate lip-synced video
    
    Uses pre-loaded models - much faster than subprocess!
    """
    try:
        audio_path = request.audio_path
        avatar_path = request.avatar_path
        output_path = request.output_path
        fps = request.fps
        
        # Validate inputs
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=400, detail=f"Audio file not found: {audio_path}")
        if not os.path.exists(avatar_path):
            raise HTTPException(status_code=400, detail=f"Avatar file not found: {avatar_path}")
        
        print(f"\n🎬 Generating video...")
        print(f"  Audio: {audio_path}")
        print(f"  Avatar: {avatar_path}")
        
        # Create temp directory
        os.makedirs('temp', exist_ok=True)
        
        # Extract audio to WAV
        subprocess.run(
            ['ffmpeg', '-y', '-i', audio_path, '-ac', '1', '-ar', '16000', '-strict', '-2', 'temp/temp.wav'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        
        # Load and process audio
        wav = audio.load_wav('temp/temp.wav', 16000)
        mel = audio.melspectrogram(wav)
        
        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise HTTPException(status_code=400, detail="Invalid audio - mel contains NaN")
        
        # Create mel chunks
        mel_step_size = 16
        mel_chunks = []
        mel_idx_multiplier = 80. / fps
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx:start_idx + mel_step_size])
            i += 1
        
        print(f"  Generated {len(mel_chunks)} mel chunks")
        
        # Load video frames
        video_stream = cv2.VideoCapture(avatar_path)
        full_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                break
            full_frames.append(frame)
        video_stream.release()
        
        if len(full_frames) == 0:
            raise HTTPException(status_code=400, detail="No frames in video")
        
        full_frames = full_frames[:len(mel_chunks)]
        orig_h, orig_w = full_frames[0].shape[:-1]
        
        # Select target face
        target_id = select_target_face(detector, recognition, full_frames[0])
        
        # Detect faces in all frames
        aligned_faces, sub_faces, matrix, face_errors = face_detect(full_frames, target_id)
        
        # Generate video
        gen = datagen(sub_faces, mel_chunks, static=False)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('temp/temp.mp4', fourcc, fps, (orig_w, orig_h))
        
        # Create face mask
        sub_face_mask = np.zeros((256, 256), dtype=np.uint8)
        sub_face_mask = cv2.rectangle(sub_face_mask, (42, 65), (214, 249), (255, 255, 255), -1)
        sub_face_mask = cv2.GaussianBlur(sub_face_mask.astype(np.uint8), (29, 29), cv2.BORDER_DEFAULT)
        sub_face_mask = cv2.cvtColor(sub_face_mask, cv2.COLOR_GRAY2RGB) / 255
        
        fc = 0
        for i, (img_batch, mel_batch, frames) in enumerate(tqdm(gen, total=len(mel_chunks), desc="Generating")):
            if fc >= len(full_frames):
                fc = 0
            
            face_err = face_errors[fc]
            
            # Prepare input
            img_batch = img_batch.transpose((0, 3, 1, 2)).astype(np.float32)
            mel_batch = mel_batch.transpose((0, 3, 1, 2)).astype(np.float32)
            
            # Run inference (models already loaded!)
            pred = wav2lip_model.run(None, {
                'mel_spectrogram': mel_batch,
                'video_frames': img_batch
            })[0][0]
            
            pred = (pred.transpose(1, 2, 0) * 255).astype(np.uint8)
            
            # Get matrices
            mat = matrix[fc]
            mat_rev = cv2.invertAffineTransform(mat)
            
            aligned_face = aligned_faces[fc]
            aligned_face_orig = aligned_face.copy()
            p_aligned = aligned_face.copy()
            full_frame = full_frames[fc]
            
            fc += 1
            
            # Resize prediction
            p = cv2.resize(pred, (132, 176))
            p_aligned[65:241, 62:194] = p
            aligned_face = (sub_face_mask * p_aligned + (1 - sub_face_mask) * aligned_face_orig).astype(np.uint8)
            
            # Warp back
            if face_err == 0:
                dealigned_face = cv2.warpAffine(aligned_face, mat_rev, (full_frame.shape[1], full_frame.shape[0]))
                static_mask = np.zeros((256, 256), dtype=np.uint8)
                static_mask = cv2.ellipse(static_mask, (112, 162), (62, 54), 0, 0, 360, (255, 255, 255), -1)
                static_mask = cv2.GaussianBlur(static_mask, (19, 19), cv2.BORDER_DEFAULT)
                static_mask = cv2.cvtColor(static_mask, cv2.COLOR_GRAY2RGB) / 255
                mask = cv2.warpAffine(static_mask, mat_rev, (full_frame.shape[1], full_frame.shape[0]))
                res = (mask * dealigned_face + (1 - mask) * full_frame).astype(np.uint8)
            else:
                res = full_frame
            
            out.write(res)
        
        out.release()
        
        # Combine audio and video
        command = [
            'ffmpeg', '-y', '-i', audio_path, '-i', 'temp/temp.mp4',
            '-shortest', '-vcodec', 'copy', '-acodec', 'libmp3lame',
            '-ac', '2', '-ar', '44100', '-ab', '128000', '-strict', '-2',
            output_path
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                      shell=platform.system() != 'Windows')
        
        # Cleanup
        if os.path.exists('temp/temp.mp4'):
            os.remove('temp/temp.mp4')
        if os.path.exists('temp/temp.wav'):
            os.remove('temp/temp.wav')
        
        print(f"✅ Video saved: {output_path}")
        
        return {
            "status": "success",
            "output_path": output_path,
            "frames": len(mel_chunks)
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check if server is running and models are loaded"""
    return {
        "status": "healthy",
        "models_loaded": wav2lip_model is not None,
        "device": device
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("Starting Wav2Lip Server on http://localhost:8000")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8000)