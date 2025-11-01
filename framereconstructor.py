
import cv2
import numpy as np
from pathlib import Path
import time
import json
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')
try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available, using numpy alternatives")

try:
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False
    print("Warning: concurrent.futures not available, using sequential processing")


class FrameReconstructor:
    """
    Advanced frame reconstruction using multiple techniques:
    1. Optical Flow Analysis
    2. Feature Matching (ORB)
    3. Structural Similarity
    4. Color Histogram Correlation
    """
    
    def __init__(self, video_path: str, output_path: str = "reconstructed_video.mp4"):
        self.video_path = video_path
        self.output_path = output_path
        self.frames = []
        self.frame_count = 0
        self.fps = 60
        self.width = 0
        self.height = 0
        
    def load_frames(self) -> bool:
        """Load all frames from the jumbled video"""
        print("Loading frames from video...")
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print("Error: Could not open video file")
            return False
        
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)
        
        cap.release()
        self.frame_count = len(self.frames)
        print(f"Loaded {self.frame_count} frames ({self.width}x{self.height} @ {self.fps}fps)")
        return True
    
    def compute_frame_features(self, frame: np.ndarray) -> dict:
        """Extract multiple features from a frame"""
        # Resize for faster processing
        small_frame = cv2.resize(frame, (320, 180))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        features = {}

        hist = cv2.calcHist([small_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        features['histogram'] = cv2.normalize(hist, hist).flatten()
        edges = cv2.Canny(gray, 50, 150)
        features['edges'] = edges
  
        orb = cv2.ORB_create(nfeatures=100)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        features['keypoints'] = keypoints
        features['descriptors'] = descriptors
        
        features['mean'] = np.mean(small_frame, axis=(0, 1))
        features['std'] = np.std(small_frame, axis=(0, 1))
        
        features['frame_small'] = small_frame
        features['gray'] = gray
        
        return features
    
    def compute_similarity(self, feat1: dict, feat2: dict) -> float:
        """Compute similarity score between two frames"""
        scores = []

        hist_corr = cv2.compareHist(
            feat1['histogram'].reshape(-1, 1).astype(np.float32),
            feat2['histogram'].reshape(-1, 1).astype(np.float32),
            cv2.HISTCMP_CORREL
        )
        scores.append(max(0, hist_corr) * 0.3)
 
        mse = np.mean((feat1['frame_small'].astype(float) - feat2['frame_small'].astype(float)) ** 2)
        if mse == 0:
            ssim_score = 1.0
        else:
            ssim_score = 1.0 / (1.0 + mse / 1000.0)
        scores.append(ssim_score * 0.4)
  
        if feat1['descriptors'] is not None and feat2['descriptors'] is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            try:
                matches = bf.match(feat1['descriptors'], feat2['descriptors'])
                match_score = len(matches) / 100.0  # Normalize by max features
                scores.append(min(match_score, 1.0) * 0.3)
            except:
                scores.append(0.0)
        else:
            scores.append(0.0)
        
        return sum(scores)
    
    def build_similarity_matrix(self, features: List[dict]) -> np.ndarray:
        """Build a similarity matrix between all frames"""
        print("Building similarity matrix...")
        n = len(features)
        similarity_matrix = np.zeros((n, n))
        

        total_comparisons = n * (n - 1) // 2
        current = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.compute_similarity(features[i], features[j])
                similarity_matrix[i][j] = sim
                similarity_matrix[j][i] = sim
                
                current += 1
                if current % 1000 == 0:
                    print(f"Progress: {current}/{total_comparisons} comparisons")
        
        return similarity_matrix
    
    def greedy_path_reconstruction(self, similarity_matrix: np.ndarray) -> List[int]:
        """
        Reconstruct frame order using greedy nearest neighbor approach
        Start from a frame and always pick the most similar unvisited frame
        """
        print("Reconstructing frame order using greedy approach...")
        n = len(similarity_matrix)

        best_path = None
        best_score = -1

        start_candidates = [0, n//4, n//2, 3*n//4, n-1]
        
        for start_idx in start_candidates:
            visited = set()
            path = [start_idx]
            visited.add(start_idx)
            current = start_idx
            total_score = 0
            
            while len(visited) < n:
                max_sim = -1
                next_frame = -1
                
                for j in range(n):
                    if j not in visited and similarity_matrix[current][j] > max_sim:
                        max_sim = similarity_matrix[current][j]
                        next_frame = j
                
                if next_frame == -1:
                    break
                
                path.append(next_frame)
                visited.add(next_frame)
                total_score += max_sim
                current = next_frame
            
            avg_score = total_score / (len(path) - 1) if len(path) > 1 else 0
            
            if avg_score > best_score:
                best_score = avg_score
                best_path = path
        
        print(f"Best path score: {best_score:.4f}")
        return best_path
    
    def optical_flow_ordering(self) -> List[int]:
        """
        Alternative approach: Use optical flow magnitude to find sequential frames
        """
        print("Using optical flow analysis...")
        n = len(self.frames)
        flow_matrix = np.zeros((n, n))
        
        for i in range(n):
            gray1 = cv2.cvtColor(cv2.resize(self.frames[i], (320, 180)), cv2.COLOR_BGR2GRAY)
            for j in range(n):
                if i == j:
                    continue
                gray2 = cv2.cvtColor(cv2.resize(self.frames[j], (320, 180)), cv2.COLOR_BGR2GRAY)
                
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
        
                magnitude = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
                flow_matrix[i][j] = -magnitude  
        
        return self.greedy_path_reconstruction(-flow_matrix)
    
    def reconstruct(self, method: str = "hybrid") -> List[int]:
        """
        Main reconstruction method
        method: 'similarity', 'optical_flow', or 'hybrid'
        """
        start_time = time.time()
        
        if not self.load_frames():
            return []
        
        if method == "optical_flow":
            frame_order = self.optical_flow_ordering()
        else:
       
            print("Extracting features from all frames...")
            features = []
            for i, frame in enumerate(self.frames):
                if i % 50 == 0:
                    print(f"Processing frame {i}/{self.frame_count}")
                features.append(self.compute_frame_features(frame))
        
            similarity_matrix = self.build_similarity_matrix(features)
    
            frame_order = self.greedy_path_reconstruction(similarity_matrix)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"\nReconstruction completed in {execution_time:.2f} seconds")
   
        log_data = {
            "execution_time_seconds": execution_time,
            "frame_count": self.frame_count,
            "method": method,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open("execution_log.json", "w") as f:
            json.dump(log_data, f, indent=2)
        
        return frame_order
    
    def save_video(self, frame_order: List[int]) -> bool:
        """Save reconstructed video"""
        print(f"\nSaving reconstructed video to {self.output_path}...")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        
        for idx in frame_order:
            out.write(self.frames[idx])
        
        out.release()
        print("Video saved successfully!")
        return True


def main():
    """Main execution function"""
    print("=" * 60)
    print("JUMBLED FRAMES RECONSTRUCTION CHALLENGE")
    print("=" * 60)
    
    INPUT_VIDEO = "jumbled_video.mp4"
    OUTPUT_VIDEO = "reconstructed_video.mp4"
    METHOD = "similarity"  

    if not Path(INPUT_VIDEO).exists():
        print(f"Error: Input video '{INPUT_VIDEO}' not found!")
        print("Please download the video and place it in the same directory as this script.")
        return

    reconstructor = FrameReconstructor(INPUT_VIDEO, OUTPUT_VIDEO)

    frame_order = reconstructor.reconstruct(method=METHOD)
    
    if not frame_order:
        print("Reconstruction failed!")
        return

    reconstructor.save_video(frame_order)
    
    print("\n" + "=" * 60)
    print("RECONSTRUCTION COMPLETE!")
    print(f"Output saved to: {OUTPUT_VIDEO}")
    print(f"Execution log saved to: execution_log.json")
    print("=" * 60)


if __name__ == "__main__":
    main()