"""
Video Inference Script
Perform segmentation on video files frame by frame
"""
import os
import sys
from pathlib import Path
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(str(Path(__file__).parent.parent))

from models import ResNet50UNet
from config import CHECKPOINTS_DIR, VIDEO_CONFIG, RESULTS_ROOT


class VideoSegmenter:
    """Video segmentation processor"""
    
    def __init__(self, model_path, device=None):
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = ResNet50UNet(in_channels=3, out_channels=1, pretrained=False)
        
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Model checkpoint not found at {model_path}")
            print("Using random initialization (results will be poor!)")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Transform
        self.transform = A.Compose([
            A.Resize(384, 384),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    
    @torch.no_grad()
    def segment_frame(self, frame):
        """
        Segment a single frame
        
        Args:
            frame: numpy array (H, W, 3) in BGR format
            
        Returns:
            mask: numpy array (H, W) with values in [0, 1]
        """
        original_h, original_w = frame.shape[:2]
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Transform
        transformed = self.transform(image=frame_rgb)
        image = transformed['image'].unsqueeze(0).to(self.device)
        
        # Inference
        output = self.model(image)
        
        # Convert to numpy and resize back
        mask = output.squeeze().cpu().numpy()
        mask = cv2.resize(mask, (original_w, original_h))
        
        return mask
    
    def process_video(self, video_path, output_path=None, 
                     overlay=True, alpha=0.5, color_map='jet'):
        """
        Process entire video
        
        Args:
            video_path: path to input video
            output_path: path to save output video
            overlay: whether to overlay mask on original frame
            alpha: overlay transparency
            color_map: colormap for mask visualization
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nVideo Info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        
        # Output path
        if output_path is None:
            output_dir = RESULTS_ROOT / 'video_inference'
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{video_path.stem}_segmented.mp4"
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )
        
        print(f"\nProcessing video...")
        pbar = tqdm(total=total_frames, desc='Segmenting frames')
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Segment frame
            mask = self.segment_frame(frame)
            
            # Create output frame
            if overlay:
                output_frame = self.create_overlay(frame, mask, alpha, color_map)
            else:
                # Show mask only
                mask_vis = (mask * 255).astype(np.uint8)
                output_frame = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
            
            # Write frame
            out.write(output_frame)
            pbar.update(1)
        
        pbar.close()
        cap.release()
        out.release()
        
        print(f"\nVideo saved to: {output_path}")
        return output_path
    
    def create_overlay(self, frame, mask, alpha=0.5, color_map='jet'):
        """
        Create overlay visualization
        
        Args:
            frame: original frame (H, W, 3) BGR
            mask: segmentation mask (H, W) values in [0, 1]
            alpha: overlay transparency
            color_map: colormap name
        """
        # Convert mask to colored heatmap
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        if color_map == 'jet':
            colored_mask = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
        elif color_map == 'hot':
            colored_mask = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_HOT)
        else:
            colored_mask = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
        
        # Create overlay
        overlay = cv2.addWeighted(frame, 1-alpha, colored_mask, alpha, 0)
        
        return overlay
    
    def process_frame_batch(self, frames):
        """Process multiple frames in batch"""
        # TODO: Implement batch processing for better performance
        pass


def main():
    parser = argparse.ArgumentParser(description='Video Segmentation Inference')
    parser.add_argument('--video_path', type=str, required=True, 
                       help='Path to input video')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save output video')
    parser.add_argument('--model_path', type=str, 
                       default=str(CHECKPOINTS_DIR / 'best_model.pth'),
                       help='Path to model checkpoint')
    parser.add_argument('--overlay', action='store_true', default=True,
                       help='Overlay mask on original frame')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Overlay transparency (0-1)')
    parser.add_argument('--color_map', type=str, default='jet',
                       choices=['jet', 'hot', 'hsv'],
                       help='Colormap for visualization')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Create segmenter
    model_path = Path(args.model_path)
    segmenter = VideoSegmenter(model_path, device=args.device)
    
    # Process video
    output_path = segmenter.process_video(
        args.video_path,
        args.output_path,
        overlay=args.overlay,
        alpha=args.alpha,
        color_map=args.color_map
    )
    
    print("\n✓ Video processing completed!")


if __name__ == '__main__':
    main()