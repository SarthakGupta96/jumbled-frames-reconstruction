"""
Testing and Evaluation Script for Frame Reconstruction
Compares reconstructed video with original (if available)
"""

import cv2
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import List, Tuple


class ReconstructionEvaluator:
    """Evaluate reconstruction quality"""
    
    def __init__(self, reconstructed_path: str, original_path: str = None):
        self.reconstructed_path = reconstructed_path
        self.original_path = original_path
        self.reconstructed_frames = []
        self.original_frames = []
        
    def load_videos(self) -> bool:
        """Load reconstructed and original videos"""
        print("Loading videos...")
        
        # Load reconstructed
        if not Path(self.reconstructed_path).exists():
            print(f"Error: {self.reconstructed_path} not found")
            return False
            
        cap = cv2.VideoCapture(self.reconstructed_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.reconstructed_frames.append(frame)
        cap.release()
        
        print(f"Loaded {len(self.reconstructed_frames)} reconstructed frames")
        
        # Load original if available
        if self.original_path and Path(self.original_path).exists():
            cap = cv2.VideoCapture(self.original_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                self.original_frames.append(frame)
            cap.release()
            print(f"Loaded {len(self.original_frames)} original frames")
        else:
            print("Original video not available - skipping accuracy metrics")
            
        return True
    
    def compute_frame_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compute SSIM between two frames"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Compute MSE
        mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
        
        # Convert to similarity score
        if mse == 0:
            return 1.0
        
        # PSNR-like score
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        # Normalize to 0-1 range
        return min(psnr / 50.0, 1.0)
    
    def evaluate_temporal_coherence(self) -> dict:
        """Evaluate temporal coherence of reconstructed video"""
        print("\nEvaluating temporal coherence...")
        
        if len(self.reconstructed_frames) < 2:
            return {}
        
        similarities = []
        for i in range(len(self.reconstructed_frames) - 1):
            sim = self.compute_frame_similarity(
                self.reconstructed_frames[i],
                self.reconstructed_frames[i + 1]
            )
            similarities.append(sim)
        
        return {
            "average_coherence": np.mean(similarities),
            "min_coherence": np.min(similarities),
            "max_coherence": np.max(similarities),
            "std_coherence": np.std(similarities),
            "coherence_scores": similarities
        }
    
    def evaluate_accuracy(self) -> dict:
        """Evaluate accuracy against original video"""
        if not self.original_frames or len(self.original_frames) != len(self.reconstructed_frames):
            return {}
        
        print("\nEvaluating accuracy against original...")
        
        n = len(self.original_frames)
        correct_positions = 0
        position_errors = []
        
        # For each reconstructed frame, find best match in original
        for i, recon_frame in enumerate(self.reconstructed_frames):
            best_match_idx = -1
            best_similarity = -1
            
            for j, orig_frame in enumerate(self.original_frames):
                sim = self.compute_frame_similarity(recon_frame, orig_frame)
                if sim > best_similarity:
                    best_similarity = sim
                    best_match_idx = j
            
            # Check if position is correct
            if best_match_idx == i:
                correct_positions += 1
            
            position_error = abs(best_match_idx - i)
            position_errors.append(position_error)
        
        accuracy = (correct_positions / n) * 100
        
        return {
            "frame_accuracy_percent": accuracy,
            "correct_positions": correct_positions,
            "total_frames": n,
            "average_position_error": np.mean(position_errors),
            "max_position_error": np.max(position_errors),
            "position_errors": position_errors
        }
    
    def visualize_results(self, coherence_data: dict, accuracy_data: dict = None):
        """Create visualizations of results"""
        print("\nGenerating visualizations...")
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Temporal Coherence
        if coherence_data and 'coherence_scores' in coherence_data:
            scores = coherence_data['coherence_scores']
            axes[0].plot(scores, linewidth=1)
            axes[0].axhline(y=np.mean(scores), color='r', linestyle='--', 
                           label=f'Average: {np.mean(scores):.3f}')
            axes[0].set_xlabel('Frame Transition')
            axes[0].set_ylabel('Similarity Score')
            axes[0].set_title('Temporal Coherence Between Consecutive Frames')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Position Errors (if available)
        if accuracy_data and 'position_errors' in accuracy_data:
            errors = accuracy_data['position_errors']
            axes[1].plot(errors, linewidth=1, color='orange')
            axes[1].axhline(y=0, color='g', linestyle='--', label='Perfect Match')
            axes[1].set_xlabel('Frame Index')
            axes[1].set_ylabel('Position Error')
            axes[1].set_title('Frame Position Errors')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'Original video not available\nfor accuracy evaluation',
                        ha='center', va='center', transform=axes[1].transAxes,
                        fontsize=12)
            axes[1].set_xticks([])
            axes[1].set_yticks([])
        
        plt.tight_layout()
        plt.savefig('evaluation_results.png', dpi=150)
        print("Saved visualization to: evaluation_results.png")
        plt.close()
    
    def generate_report(self) -> dict:
        """Generate complete evaluation report"""
        print("\n" + "="*60)
        print("RECONSTRUCTION EVALUATION REPORT")
        print("="*60)
        
        # Load videos
        if not self.load_videos():
            return {}
        
        # Evaluate temporal coherence
        coherence_data = self.evaluate_temporal_coherence()
        
        # Evaluate accuracy (if original available)
        accuracy_data = self.evaluate_accuracy()
        
        # Create visualizations
        self.visualize_results(coherence_data, accuracy_data)
        
        # Compile report
        report = {
            "reconstructed_video": self.reconstructed_path,
            "original_video": self.original_path,
            "frame_count": len(self.reconstructed_frames),
            "temporal_coherence": coherence_data,
            "accuracy_metrics": accuracy_data
        }
        
        # Print summary
        print("\n--- Temporal Coherence ---")
        if coherence_data:
            print(f"Average Coherence: {coherence_data.get('average_coherence', 0):.4f}")
            print(f"Min Coherence: {coherence_data.get('min_coherence', 0):.4f}")
            print(f"Max Coherence: {coherence_data.get('max_coherence', 0):.4f}")
        
        if accuracy_data:
            print("\n--- Accuracy Metrics ---")
            print(f"Frame Accuracy: {accuracy_data.get('frame_accuracy_percent', 0):.2f}%")
            print(f"Correct Positions: {accuracy_data.get('correct_positions', 0)}/{accuracy_data.get('total_frames', 0)}")
            print(f"Average Position Error: {accuracy_data.get('average_position_error', 0):.2f} frames")
        
        print("\n" + "="*60)
        
        # Save report
        with open('evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        print("\nFull report saved to: evaluation_report.json")
        
        return report


def main():
    """Main testing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate reconstructed video')
    parser.add_argument('--reconstructed', default='reconstructed_video.mp4',
                       help='Path to reconstructed video')
    parser.add_argument('--original', default=None,
                       help='Path to original video (if available)')
    
    args = parser.parse_args()
    
    evaluator = ReconstructionEvaluator(args.reconstructed, args.original)
    evaluator.generate_report()


if __name__ == "__main__":
    main()