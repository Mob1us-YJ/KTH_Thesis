"""
WiSe4Car Video Annotation Tool - Core Code

Features:
1. Play video with manual annotation
2. Auto-detect motion segments (semi-automatic annotation helper)
3. Export annotations as JSON format
4. Display real-time statistics

Usage:
  python video_annotation_tool.py --video path/to/video.avi
"""

import cv2
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import numpy as np


class VideoAnnotationTool:
    """Interactive video annotation tool"""
    
    def __init__(self, video_path: str, behavior_labels: Optional[Dict] = None):
        """
        Initialize annotation tool
        
        Args:
            video_path: Path to video file
            behavior_labels: Behavior label definitions (dict)
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_seconds = self.total_frames / self.fps
        
        # Default behavior labels
        self.behavior_labels = behavior_labels or {
            1: 'sitting',
            2: 'reaching',
            3: 'turning',
            4: 'bending',
            5: 'waving',
            6: 'talking',
            7: 'using_phone',
            8: 'leaving',
            9: 'moving_around',
            0: 'no_motion',
        }
        
        self.annotations = []
        self.current_frame = 0
        self.paused = False
        self.current_action = None
        self.segment_start_frame = None
        
        print(f"\n{'='*70}")
        print(f"Video Annotation Tool - {self.video_path.name}")
        print(f"{'='*70}")
        print(f"Total frames: {self.total_frames}")
        print(f"FPS: {self.fps:.2f}")
        print(f"Duration: {self.duration_seconds:.2f} seconds ({int(self.duration_seconds//60)}:{int(self.duration_seconds%60):02d})")
        print(f"{'='*70}\n")
        
    def _frame_to_seconds(self, frame_num: int) -> float:
        """Convert frame number to seconds"""
        return frame_num / self.fps
    
    def _seconds_to_frame(self, seconds: float) -> int:
        """Convert seconds to frame number"""
        return int(seconds * self.fps)
    
    def _frame_to_mmss(self, frame_num: int) -> str:
        """Convert frame number to MM:SS format"""
        seconds = self._frame_to_seconds(frame_num)
        mins = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{mins:02d}:{secs:02d}"
    
    def print_help(self):
        """Print help information"""
        print("\n" + "="*70)
        print("Keyboard Controls")
        print("="*70)
        print("  SPACE       - Pause/Resume")
        print("  J           - Go back 5 frames")
        print("  L           - Go forward 5 frames")
        print("  I           - Go forward 1 second")
        print("  K           - Go back 1 second")
        print()
        print("  Mark actions (press corresponding number key):")
        for key, label in self.behavior_labels.items():
            print(f"    {key} - {label.upper()}")
        print()
        print("  S - Mark segment start")
        print("  E - Mark segment end")
        print("  C - Remove last annotation")
        print("  H - Show this help")
        print("  Q - Exit and save")
        print("="*70 + "\n")
    
    def start_interactive_annotation(self):
        """Start interactive annotation"""
        self.print_help()
        
        window_name = f"Annotation Tool - {self.video_path.name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1000, 700)
        
        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("\nVideo playback finished.")
                    break
                
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
            
            # Draw information
            frame = self._draw_info(frame)
            
            cv2.imshow(window_name, frame)
            
            # Process key input
            key = cv2.waitKey(30)
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.paused = not self.paused
            elif key == ord('i'):  # I - Go forward 1 second
                self.current_frame = min(self.total_frames - 1, 
                                        self.current_frame + int(self.fps))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                self.paused = True
            elif key == ord('k'):  # K - Go back 1 second
                self.current_frame = max(0, self.current_frame - int(self.fps))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                self.paused = True
            elif key == ord('j'):  # J - Go back 5 frames
                self.current_frame = max(0, self.current_frame - 5)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                self.paused = True
            elif key == ord('l'):  # L - Go forward 5 frames
                self.current_frame = min(self.total_frames - 1, 
                                        self.current_frame + 5)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                self.paused = True
            elif key == ord('s'):
                self.segment_start_frame = self.current_frame
                print(f"Segment start marked: {self._frame_to_mmss(self.current_frame)}")
            elif key == ord('e') and self.segment_start_frame is not None:
                end_frame = self.current_frame
                if self.current_action:
                    self.add_annotation(
                        self.segment_start_frame,
                        end_frame,
                        self.current_action
                    )
                    self.segment_start_frame = None
            elif key == ord('c'):
                if self.annotations:
                    removed = self.annotations.pop()
                    print(f"Removed last annotation: {removed['action']} "
                          f"({self._frame_to_mmss(removed['start_frame'])} - "
                          f"{self._frame_to_mmss(removed['end_frame'])})")
            elif key == ord('h'):
                self.print_help()
            elif 48 <= key <= 57:  # ASCII codes for 0-9
                action_key = key - 48
                if action_key in self.behavior_labels:
                    self.current_action = self.behavior_labels[action_key]
                    print(f"Selected action: {self.current_action.upper()}")
        
        cv2.destroyAllWindows()
        self.cap.release()
    
    def add_annotation(self, start_frame: int, end_frame: int, action: str, 
                       confidence: float = 1.0, notes: str = ""):
        """Add annotation"""
        if start_frame >= end_frame:
            print("Error: Start frame must be before end frame")
            return
        
        annotation = {
            "id": len(self.annotations) + 1,
            "start_time": self._frame_to_seconds(start_frame),
            "end_time": self._frame_to_seconds(end_frame),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "action": action,
            "confidence": confidence,
            "notes": notes
        }
        
        self.annotations.append(annotation)
        
        print(f"Added annotation [{len(self.annotations)}]: "
              f"{action} ({self._frame_to_mmss(start_frame)} - "
              f"{self._frame_to_mmss(end_frame)})")
    
    def _draw_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw information on frame"""
        h, w = frame.shape[:2]
        
        # Draw background bars
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, h-40), (w, h), (0, 0, 0), -1)
        
        # Time information
        current_time = self._frame_to_mmss(self.current_frame)
        total_time = self._frame_to_mmss(self.total_frames - 1)
        status = "PAUSED" if self.paused else "PLAYING"
        
        text = f"{status} | Time: {current_time} / {total_time} | "
        text += f"Frame: {self.current_frame}/{self.total_frames}"
        
        cv2.putText(frame, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Current selected action
        if self.current_action:
            action_text = f"Current action: {self.current_action.upper()}"
            cv2.putText(frame, action_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display annotations list in top-right corner
        if self.annotations:
            right_margin = 10
            line_height = 25
            y_start = 20
            
            anno_header = f"Annotation list({len(self.annotations)})"
            anno_header_size = cv2.getTextSize(anno_header, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            x_pos = w - anno_header_size[0] - right_margin
            
            cv2.putText(frame, anno_header, (x_pos, y_start),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            
            # Show last 3 annotations
            num_display = min(3, len(self.annotations))
            start_idx = max(0, len(self.annotations) - num_display)
            
            for i, anno in enumerate(self.annotations[start_idx:]):
                y_pos = y_start + (i + 1) * line_height
                anno_text = f"[{len(self.annotations) - num_display + i + 1}] "
                anno_text += f"{anno['action'].upper()}: "
                anno_text += f"{self._frame_to_mmss(anno['start_frame'])} -{self._frame_to_mmss(anno['end_frame'])}"
                
                # Color based on action type
                color = (0, 200, 255)  # Default cyan
                if anno['action'] in ['reaching', 'waving', 'moving_around']:
                    color = (0, 255, 0)  # Green for movement
                elif anno['action'] in ['sitting', 'turning', 'bending']:
                    color = (255, 0, 0)  # Blue for position change
                elif anno['action'] == 'no_motion':
                    color = (100, 100, 100)  # Gray
                
                text_size = cv2.getTextSize(anno_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                x_pos = w - text_size[0] - right_margin
                
                cv2.putText(frame, anno_text, (x_pos, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Progress bar
        progress = self.current_frame / self.total_frames
        bar_width = 300
        cv2.rectangle(frame, (10, h-25), (10 + int(bar_width * progress), h-15),
                     (0, 255, 0), -1)
        cv2.rectangle(frame, (10, h-25), (10 + bar_width, h-15),
                     (100, 100, 100), 2)
        
        # Annotation statistics
        anno_text = f"Annotations: {len(self.annotations)}"
        cv2.putText(frame, anno_text, (w - 300, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return frame
    
    def save_annotations(self, output_json: str):
        """Save annotations as JSON"""
        output_path = Path(output_json)
        
        data = {
            "session_id": self.video_path.stem,
            "video_file": self.video_path.name,
            "video_duration_seconds": self.duration_seconds,
            "video_fps": self.fps,
            "total_frames": self.total_frames,
            "annotation_count": len(self.annotations),
            "annotations": self.annotations,
            "annotator": "manual",
            "annotation_date": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\nAnnotations saved: {output_path}")
        print(f"  - Total annotations: {len(self.annotations)}")
        
        # Print summary
        action_counts = {}
        for anno in self.annotations:
            action = anno['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        print(f"  - Action distribution:")
        for action, count in sorted(action_counts.items()):
            print(f"    * {action}: {count} segments")
    
    def load_annotations_from_json(self, json_path: str):
        """Load existing annotations from JSON"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.annotations = data.get('annotations', [])
        print(f"Loaded {len(self.annotations)} existing annotations")


class MotionDetectionHelper:
    """Motion detection helper tool (for semi-automatic annotation)"""
    
    def __init__(self, video_path: str, threshold: float = 0.005):
        """
        Initialize motion detector
        
        Args:
            video_path: Path to video file
            threshold: Motion detection threshold (0-1), lowered to 0.005 for small motions
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))
        self.threshold = threshold
        self.motion_frames = []
    
    def detect_motion_segments(self, min_duration_frames: int = 5) -> List[Tuple]:
        """
        Detect motion segments in video using aggressive detection for small motions
        
        Returns:
            List of (start_frame, end_frame) tuples
        """
        print(f"Detecting motion segments (threshold: {self.threshold})...")
        
        prev_frame = None
        motion_indices = []
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)  # Reduced blur for finer details
            
            if prev_frame is not None:
                # Method 1: Frame difference (detects overall motion)
                diff = cv2.absdiff(gray, prev_frame)
                motion_overall = diff.mean() / 255.0
                
                # Method 2: Robust detection using percentile threshold
                # Check if top pixels have significant changes (catches local motions)
                diff_values = diff.flatten()
                percentile_90 = np.percentile(diff_values, 90)
                
                # Method 3: Edge/corner-focused detection
                # Focus on regions with edges where motion is more visible
                diff_threshold = 8  # Very low for sensitivity to small changes
                significant_changes = np.sum(np.abs(diff.astype(float)) > diff_threshold)
                motion_edges = significant_changes / (gray.shape[0] * gray.shape[1])
                
                # AGGRESSIVE detection: trigger if ANY method detects motion
                # This should catch head turns, reaching, and other local motions
                detect_overall = motion_overall > (self.threshold * 0.8)
                detect_percentile = percentile_90 > (self.threshold * 5)  # High percentile indicates local motion
                detect_edges = motion_edges > (self.threshold * 0.5)
                
                if detect_overall or detect_percentile or detect_edges:
                    motion_indices.append(frame_count)
            
            prev_frame = gray
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"  Processing: {frame_count} frames...")
        
        self.cap.release()
        
        # Merge consecutive motion frames with smaller gap tolerance for better continuity
        segments = self._merge_consecutive_frames(motion_indices, min_duration_frames)
        
        print(f"Detected {len(segments)} motion segments")
        for i, (start, end) in enumerate(segments, 1):
            print(f"  [{i}] Frame {start}-{end} ({end-start} frames)")
        
        return segments
    
    @staticmethod
    def _merge_consecutive_frames(indices: List[int], min_duration: int) -> List[Tuple]:
        """Merge consecutive frame indices into segments"""
        if not indices:
            return []
        
        segments = []
        start = indices[0]
        prev = indices[0]
        
        for idx in indices[1:]:
            if idx - prev > 1:
                if prev - start >= min_duration:
                    segments.append((start, prev))
                start = idx
            prev = idx
        
        if prev - start >= min_duration:
            segments.append((start, prev))
        
        return segments


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="WiSe4Car Video Annotation Tool"
    )
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path (default: video_annotations.json)')
    parser.add_argument('--motion-detect', action='store_true',
                       help='Detect motion segments first (assist annotation)')
    
    args = parser.parse_args()
    
    # Initialize annotation tool
    tool = VideoAnnotationTool(args.video)
    
    # Run motion detection if specified
    if args.motion_detect:
        detector = MotionDetectionHelper(args.video)
        detector.detect_motion_segments()
    
    # Start interactive annotation
    tool.start_interactive_annotation()
    
    # Save annotations
    output_path = args.output or f"{Path(args.video).stem}_annotations.json"
    tool.save_annotations(output_path)


if __name__ == "__main__":
    main()
