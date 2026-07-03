import cv2
import numpy as np
import argparse
from tqdm import tqdm

def apply_chromatic_aberration(frame, intensity):
    """Shifts color channels to create a classic RGB split glitch."""
    if intensity == 0:
        return frame
    
    b, g, r = cv2.split(frame)
    shift = int(intensity * 5)
    
    # Roll channels horizontally to create the split
    b = np.roll(b, shift, axis=1)
    r = np.roll(r, -shift, axis=1)
    
    return cv2.merge((b, g, r))

def apply_p_frame_drag(frame, prev_frame, intensity):
    """Simulates a datamosh P-frame drag by freezing pixels where motion is low

    or bleeding pixels forward based on frame differences.
    """
    if prev_frame is None or intensity == 0:
        return frame.copy()
    
    # Calculate absolute difference between current and previous frame
    diff = cv2.absdiff(frame, prev_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Create a mask where things are NOT moving much (or reverse based on aesthetic preference)
    # Higher intensity means more of the old frame's pixels get dragged/frozen
    threshold_val = int(255 * (1.0 - (intensity * 0.8)))
    _, motion_mask = cv2.threshold(gray_diff, threshold_val, 255, cv2.THRESH_BINARY_INV)
    
    # Dilate mask to make the pixel bleed blockier and more "compressed"
    kernel = np.ones((5, 5), np.uint8)
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)
    
    # Convert mask to 3 channels
    motion_mask_3d = cv2.merge([motion_mask, motion_mask, motion_mask])
    
    # Where mask is active, keep the previous frame's pixels (pixel dragging)
    output = np.where(motion_mask_3d == 255, prev_frame, frame)
    return output

def apply_scanline_glitch(frame, intensity):
    """Randomly displaces horizontal slices of the video for a corrupted signal look."""
    if intensity == 0:
        return frame
    
    h, w, _ = frame.shape
    num_slices = int(intensity * 15)
    out_frame = frame.copy()
    
    for _ in range(num_slices):
        slice_h = np.random.randint(5, int(h * 0.1))
        y = np.random.randint(0, h - slice_h)
        shift = np.random.randint(-int(w * 0.1 * intensity), int(w * 0.1 * intensity))
        
        # Displace the horizontal block
        out_frame[y:y+slice_h, :] = np.roll(out_frame[y:y+slice_h, :], shift, axis=1)
        
    return out_frame

def process_video(input_path, output_path, p_frame_int, color_int, scan_int):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    # Get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    prev_frame = None

    print(f"Processing video with effects -> P-Frame: {p_frame_int}, Color: {color_int}, Scanline: {scan_int}")
    
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Apply Datamosh / P-Frame drag
        glitched_frame = apply_p_frame_drag(frame, prev_frame, p_frame_int)
        
        # 2. Apply Avant-garde scanline corruption
        glitched_frame = apply_scanline_glitch(glitched_frame, scan_int)
        
        # 3. Apply Chromatic Color Aberration
        glitched_frame = apply_chromatic_aberration(glitched_frame, color_int)
        
        # Save the current output frame to become the history for the next frame's datamosh
        prev_frame = glitched_frame.copy()
        
        out.write(glitched_frame)

    cap.release()
    out.release()
    print(f"Glitch art successfully rendered to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avant-Garde Datamosh & Glitch Video Generator")
    parser.add_argument("-i", "--input", required=True, help="Path to input video file")
    parser.add_argument("-o", "--output", default="output_glitch.mp4", help="Path to save output video")
    
    # Customization intensities (0.0 to 1.0)
    parser.add_argument("--pframe", type=float, default=0.5, help="P-frame pixel dragging intensity (0.0 - 1.0)")
    parser.add_argument("--color", type=float, default=0.4, help="Color shift / Chromatic aberration intensity (0.0 - 1.0)")
    parser.add_argument("--scanline", type=float, default=0.3, help="Horizontal slice corruption intensity (0.0 - 1.0)")
    
    args = parser.parse_args()
    
    process_video(
        input_path=args.input,
        output_path=args.output,
        p_frame_int=max(0.0, min(1.0, args.pframe)),
        color_int=max(0.0, min(1.0, args.color)),
        scan_int=max(0.0, min(1.0, args.scanline))
    )
