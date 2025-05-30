#!/usr/bin/env python3
"""
Final corrected version of plot_results.py
Handles:
- Mismatched array lengths
- HH:MM:SS time format 
- Failed experiments
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
PROJECT_ROOT = Path("/Users/a11/Desktop/MT/exercise/MTexercise04/mt-exercise-4")
RESULTS_FILE = PROJECT_ROOT / "beam_results/all_results.txt"
OUTPUT_DIR = PROJECT_ROOT / "beam_results"

# Styling
plt.style.use('seaborn-v0_8')
plt.rcParams['font.family'] = 'DejaVu Sans'
COLORS = {
    'bleu': '#4C72B0',
    'time': '#DD8452',
    'failed': '#C44E52'
}

def parse_time(time_str):
    """Convert 00h:04m:18s format to seconds"""
    if "FAILED" in time_str:
        return np.nan
    h, m, s = time_str.split(':')
    return int(h.replace('h',''))*3600 + int(m.replace('m',''))*60 + float(s.replace('s',''))

def load_results():
    """Load experiment results with robust error handling"""
    beams, times, bleus = [], [], []
    with open(RESULTS_FILE) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                try:
                    beams.append(int(parts[0]))
                    times.append(parse_time(parts[1]))
                    # Handle both float and FAILED cases
                    try:
                        bleus.append(float(parts[2]))
                    except ValueError:
                        bleus.append(np.nan)  # For FAILED entries
                except ValueError:
                    continue
    return np.array(beams), np.array(times), np.array(bleus)

def plot_bleu_vs_beam(beams, bleus):
    """Plot BLEU scores with failure markers"""
    plt.figure(figsize=(10, 6))
    
    # Valid data points
    valid = ~np.isnan(bleus)
    if sum(valid) > 0:
        plt.semilogx(beams[valid], bleus[valid], 
                    'o-', color=COLORS['bleu'],
                    linewidth=2, markersize=8,
                    label='BLEU Score')
    
    # Failed runs
    failed = np.isnan(bleus)
    if any(failed):
        y_min = np.nanmin(bleus) if any(valid) else 0
        plt.scatter(beams[failed], [y_min*0.95]*sum(failed),
                    marker='x', color=COLORS['failed'],
                    s=100, linewidths=2,
                    label='Failed Runs')
    
    plt.xlabel('Beam Size (log scale)', fontsize=12)
    plt.ylabel('BLEU Score', fontsize=12)
    plt.title('BLEU Score vs Beam Size', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.savefig(OUTPUT_DIR/"bleu_vs_beam.png", dpi=300, bbox_inches='tight')

def plot_time_vs_beam(beams, times):
    """Plot translation times with trendline"""
    plt.figure(figsize=(10, 6))
    
    valid = ~np.isnan(times)
    if sum(valid) > 0:
        plt.loglog(beams[valid], times[valid], 
                  's-', color=COLORS['time'],
                  linewidth=2, markersize=8,
                  label='Translation Time')
        
        # Add trendline if enough points
        if sum(valid) > 2:
            x = np.log(beams[valid])
            y = np.log(times[valid])
            coeffs = np.polyfit(x, y, 1)
            trend = np.exp(coeffs[1]) * (beams[valid] ** coeffs[0])
            plt.plot(beams[valid], trend, '--', 
                     color='gray', alpha=0.7,
                     label=f'O(n^{coeffs[0]:.2f}) trend')
    
    plt.xlabel('Beam Size (log scale)', fontsize=12)
    plt.ylabel('Time (seconds, log scale)', fontsize=12)
    plt.title('Translation Time vs Beam Size', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.savefig(OUTPUT_DIR/"time_vs_beam.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    beams, times, bleus = load_results()
    
    print(f"Loaded data - Beams: {len(beams)}, Times: {len(times)}, BLEUs: {len(bleus)}")
    print("Beam sizes:", beams)
    print("Times (sec):", times)
    print("BLEU scores:", bleus)
    
    if len(beams) == 0:
        print("Error: No valid data loaded")
        exit(1)
    
    plot_bleu_vs_beam(beams, bleus)
    plot_time_vs_beam(beams, times)
    
    print("\nPlots saved to:")
    print(f"  - {OUTPUT_DIR/'bleu_vs_beam.png'}")
    print(f"  - {OUTPUT_DIR/'time_vs_beam.png'}")