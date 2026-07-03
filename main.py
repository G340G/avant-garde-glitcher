import cv2
import numpy as np
import argparse
import os
import sys
import random
from tqdm import tqdm

def apply_chromatic_aberration(frame, intensity):
    """Analog color fringing. Dynamically varies by frame for a swimming effect."""
    if intensity == 0:
        return frame
    
    b, g, r = cv2.split(frame)
    # Randomly jitter the shift amount slightly per frame to mimic tape instability
    dynamic_intensity = intensity * random.uniform(0.5, 1.5)
    shift = int(dynamic_intensity * 12)
    
    if shift > 0:
        b = np.roll(b, shift, axis=1)
        r = np.roll(r, -shift, axis=0) # Shift vertically for a more chaotic analog phase split
    
    return cv2.merge((b, g, r))

def apply_p_frame_drag(frame, prev_frame, intensity, is_glitch_burst):
    """Simulates deep compression generation loss and datamosh macroblock bleeding."""
    if prev_frame is None or intensity == 0:
        return frame.copy()
    
    # If we are in a burst, we aggressively drag pixels. If not, we let the video breathe.
    mult = 1.8 if is_glitch_burst else 0.4
    effective_intensity = max(0.0, min(1.0, intensity * mult))
    
    diff = cv2.absdiff(frame, prev_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    threshold_val = int(255 * (1.0 - (effective_intensity * 0.9)))
    _, motion_mask = cv2.threshold(gray_diff, threshold_val, 255, cv2.THRESH_BINARY_INV)
    
    # Avant-garde upgrade: Use a very large blocky kernel to simulate H.264 macroblocks (16x16 squares)
    kernel_size = random.choice([8, 16, 24]) if is_glitch_burst else 8
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)
    
    motion_mask_3d = cv2.merge([motion_mask, motion_mask, motion_mask])
    return np.where(motion_mask_3d == 255, prev_frame, frame)

def apply_vhs_tracking_and_macroblocks(frame, intensity, is_glitch_burst):
    """Simulates modern video generational compression loss combined with old tape tracking tears."""
    if intensity == 0:
        return frame
    
    h, w, c = frame.shape
    out_frame = frame.copy()
    
    # 1. Macroblock compression simulation (10,000 reuploads artifact)
    # Downscale and upscale roughly to simulate heavy sub-sampling compression blockiness
    if is_glitch_burst or random.random() < 0.15:
        compress_factor = random.uniform(4, 12 if is_glitch_burst else 6)
        small = cv2.resize(out_frame, (int(w // compress_factor), int(h // compress_factor)), interpolation=cv2.INTER_NEAREST)
        out_frame = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # 2. VHS Horizontal Sync Tear (Only hits specific horizontal bands)
    num_tears = int(intensity * 8) if is_glitch_burst else int(intensity * 2)
    for _ in range(num_tears):
        band_h = np.random.randint(4, int(h * 0.08) if not is_glitch_burst else int(h * 0.25))
        y = np.random.randint(0, h - band_h)
        
        # Pull or push pixels horizontally with a heavy analog smear
        shift = np.random.randint(-int(w * 0.2 * intensity), int(w * 0.2 * intensity))
        out_frame[y:y+band_h, :] = np.roll(out_frame[y:y+band_h, :], shift, axis=1)
        
        # Color distortion inside the torn band (analog luma stretch)
        if random.random() < 0.5:
            out_frame[y:y+band_h, :] = cv2.add(out_frame[y:y+band_h, :], (random.randint(-40, 40), random.randint(-20, 20), random.randint(-20, 40), 0))

    return out_frame

def process_video(input_path, output_path, p_frame_int, color_int, glitch_freq):
    if not os.path.exists(input_path):
        print(f"❌ Error: The input file '{input_path}' was not found.")
        sys.exit(1)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file '{input_path}'.")
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    prev_frame = None
    
    # State machine values to control "Rhythm" instead of constant static noise
    burst_counter = 0
    is_glitch_burst = False

    print(f"🎬 Initializing Dynamic Glitch Engine...")
    
    for _ in tqdm(range(total_frames), desc="Rendering Glitch Art"):
        ret, frame = cap.read()
        if not ret:
            break
            
        # --- RHYTHM CONTROLLER ---
        # Instead of glitching equally, we cycle between "clean-ish" and "shattered"
        if burst_counter <= 0:
            # Decide if we trip a catastrophic burst based on glitch_freq
            is_glitch_burst = random.random() < (glitch_freq * 0.25)
            # A burst lasts between 4 and 15 frames. Calm periods last 10 to 30 frames.
            burst_counter = random.randint(4, 15) if is_glitch_burst else random.randint(10, 30)
        else:
            burst_counter -= 1
            
        # 1. Blocky Compression / VHS Tracking Tears
        glitched_frame = apply_vhs_tracking_and_macroblocks(frame, p_frame_int, is_glitch_burst)
        
        # 2. Datamosh / Color bleeding (fed by previous frame)
        glitched_frame = apply_p_frame_drag(glitched_frame, prev_frame, p_frame_int, is_glitch_burst)
        
        # 3. Dynamic Organic Chromatic Split
        glitched_frame = apply_chromatic_aberration(glitched_frame, color_int)
        
        # Save historical layout
        prev_frame = glitched_frame.copy()
        out.write(glitched_frame)

    cap.release()
    out.release()
    print(f"🎉 Success! Avant-garde render completed: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic Generational Decay & VHS Glitch Studio")
    parser.add_argument("-i", "--input", required=True, help="Input video")
    parser.add_argument("-o", "--output", default="glitched_output.mp4", help="Output video")
    
    # Intuitively re-labeled arguments to match the user request
    parser.add_argument("--pframe", type=float, default=0.5, help="Datamosh & macroblock corruption (0.0 - 1.0)")
    parser.add_argument("--color", type=float, default=0.4, help="Analog color separation depth (0.0 - 1.0)")
    parser.add_argument("--frequency", type=float, default=0.5, help="Glitch rhythm frequency/chaos bursts (0.0 - 1.0)")
    
    args = parser.parse_args()
    
    process_video(
        input_path=args.input,
        output_path=args.output,
        p_frame_int=max(0.0, min(1.0, args.pframe)),
        color_int=max(0.0, min(1.0, args.color)),
        glitch_freq=max(0.0, min(1.0, args.frequency))
    )
