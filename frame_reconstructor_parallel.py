
import cv2
import numpy as np
from pathlib import Path
import time
import json
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

from multiprocessing import Pool, cpu_count
from functools import partial

print(f"Available CPU cores: {cpu_count()}")


class ParallelFrameReconstructor:
    
    
    def __init__(self, video_path: str, output_path: str = "reconstructed_video.mp4", n_workers: int = None):
        self.video_path = video_path
        self.output_path = output_path
        self.frames = []
        self.frame_count = 0
        self.fps = 60
        self.width = 0
        self.height = 0
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        
    def load_frames(self) -> bool:
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
        print(f"Using {self.n_workers} parallel workers")
        return True
    
    def compute_frame_features(self, frame: np.ndarray) -> dict:
        small_frame = cv2.resize(frame, (320, 180))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        features = {}
        

        hist = cv2.calcHist([small_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        features['histogram'] = cv2.normalize(hist, hist).flatten()
   
        orb = cv2.ORB_create(nfeatures=100)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        features['keypoints'] = keypoints
        features['descriptors'] = descriptors
 
        features['frame_small'] = small_frame
        features['gray'] = gray
        
        return features
    
    def extract_all_features(self) -> List[dict]:
        """Extract features from all frames in parallel"""
        print("Extracting features from all frames in parallel...")
        
        with Pool(self.n_workers) as pool:
            features = pool.map(self.compute_frame_features, self.frames)
        
        print(f"Feature extraction complete!")
        return features
    
    @staticmethod
    def compute_similarity_pair(args) -> Tuple[int, int, float]:
        """Compute similarity between a pair of frames (for parallel processing)"""
        i, j, feat1, feat2 = args
        
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
                match_score = len(matches) / 100.0
                scores.append(min(match_score, 1.0) * 0.3)
            except:
                scores.append(0.0)
        else:
            scores.append(0.0)
        
        return (i, j, sum(scores))
    
    def build_similarity_matrix_parallel(self, features: List[dict]) -> np.ndarray:
        """Build similarity matrix using parallel processing"""
        print("Building similarity matrix in parallel...")
        n = len(features)
        similarity_matrix = np.zeros((n, n))
        
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j, features[i], features[j]))
        
        total_comparisons = len(pairs)
        print(f"Computing {total_comparisons} similarity scores...")
        
        batch_size = 1000
        completed = 0
        
        with Pool(self.n_workers) as pool:
            for batch_start in range(0, total_comparisons, batch_size):
                batch_end = min(batch_start + batch_size, total_comparisons)
                batch = pairs[batch_start:batch_end]
                
                results = pool.map(self.compute_similarity_pair, batch)
                
                for i, j, sim in results:
                    similarity_matrix[i][j] = sim
                    similarity_matrix[j][i] = sim
                
                completed += len(results)
                print(f"Progress: {completed}/{total_comparisons} comparisons ({100*completed/total_comparisons:.1f}%)")
        
        return similarity_matrix
    
    def greedy_path_reconstruction(self, similarity_matrix: np.ndarray) -> List[int]:
        """Reconstruct frame order using greedy nearest neighbor approach"""
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
                print(f"  Start {start_idx}: score = {avg_score:.4f}")
        
        print(f"\nBest path score: {best_score:.4f}")
        return best_path
    
    def reconstruct(self) -> List[int]:
        """Main reconstruction method"""
        start_time = time.time()
        
        if not self.load_frames():
            return []
        
        features = self.extract_all_features()
        
        similarity_matrix = self.build_similarity_matrix_parallel(features)
        
        frame_order = self.greedy_path_reconstruction(similarity_matrix)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"\nReconstruction completed in {execution_time:.2f} seconds")
        
        log_data = {
            "execution_time_seconds": execution_time,
            "frame_count": self.frame_count,
            "method": "parallel_similarity",
            "n_workers": self.n_workers,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open("execution_log.json", "w") as f:
            json.dump(log_data, f, indent=2)
        
        print(f"Speedup from parallelization: ~{self.n_workers}x theoretical")
        
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
    print("JUMBLED FRAMES RECONSTRUCTION - PARALLEL VERSION")
    print("=" * 60)
    
    INPUT_VIDEO = "jumbled_video.mp4"
    OUTPUT_VIDEO = "reconstructed_video.mp4"
    
    if not Path(INPUT_VIDEO).exists():
        print(f"Error: Input video '{INPUT_VIDEO}' not found!")
        print("Please download the video and place it in the same directory.")
        return
    
    reconstructor = ParallelFrameReconstructor(INPUT_VIDEO, OUTPUT_VIDEO, n_workers=None)
    
    frame_order = reconstructor.reconstruct()
    
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