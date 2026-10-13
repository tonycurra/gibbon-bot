#!/usr/bin/env python3
"""
Evaluation Tool for Agent Control Runs
======================================

Run this AFTER testing a control script. It will:
1. Show you the latest log file
2. Display key metrics
3. Generate visualization plots
4. Help you understand what worked and what didn't

Usage:
    python3 evaluate.py              # Evaluate latest run
    python3 evaluate.py --all        # Show all runs
    python3 evaluate.py --mode CIRCLE # Filter by mode
    python3 evaluate.py logs/agent_run_*.csv  # Specific files
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import glob

# ============================================================================
# FIND LOG FILES
# ============================================================================

def find_logs(pattern="logs/agent_run_*.csv", all_runs=False, mode_filter=None):
    """Find matching log files."""
    log_dir = Path("logs")
    if not log_dir.exists():
        print("No logs directory found. Run agent_control.py first.")
        return []
    
    if pattern.startswith("logs/"):
        files = sorted(glob.glob(pattern), reverse=True)
    else:
        files = sorted(glob.glob(f"logs/{pattern}"), reverse=True)
    
    if not files:
        print(f"No log files found matching: {pattern}")
        return []
    
    if not all_runs:
        files = files[:1]  # Just the latest
    
    if mode_filter:
        files = [f for f in files if mode_filter.upper() in f.upper()]
    
    return files

# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(df):
    """Calculate evaluation metrics from a run."""
    metrics = {}
    
    # Basic info
    metrics['duration'] = df['time'].max()
    metrics['steps'] = len(df)
    metrics['fps'] = len(df) / metrics['duration'] if metrics['duration'] > 0 else 0
    
    # Tracking error (FL gripper to target)
    metrics['error_mean'] = df['error_norm'].mean()
    metrics['error_max'] = df['error_norm'].max()
    metrics['error_min'] = df['error_norm'].min()
    metrics['error_std'] = df['error_norm'].std()
    metrics['error_final'] = df['error_norm'].iloc[-1]
    
    # Path following: how much of the path was covered
    if 'target_x' in df.columns:
        target_positions = df[['target_x', 'target_y', 'target_z']].values
        fl_positions = df[['fl_x', 'fl_y', 'fl_z']].values
        
        # Calculate path lengths
        target_path_length = 0
        fl_path_length = 0
        for i in range(1, len(df)):
            target_dx = target_positions[i] - target_positions[i-1]
            fl_dx = fl_positions[i] - fl_positions[i-1]
            target_path_length += np.linalg.norm(target_dx)
            fl_path_length += np.linalg.norm(fl_dx)
        
        metrics['target_path_length'] = target_path_length
        metrics['fl_path_length'] = fl_path_length
        
        # Path similarity (0-1, higher is better)
        if target_path_length > 0:
            metrics['path_fidelity'] = fl_path_length / target_path_length
        else:
            metrics['path_fidelity'] = 0
    
    # Control effort
    control_cols = [c for c in df.columns if c.startswith('ctrl_')]
    if control_cols:
        control_effort = df[control_cols].abs().sum(axis=1).mean()
        metrics['control_effort_mean'] = control_effort
        metrics['control_effort_max'] = df[control_cols].abs().max().max()
    
    # Torso movement (should be minimal for good control)
    if 'torso_x' in df.columns:
        torso_movement = 0
        for i in range(1, len(df)):
            dx = df['torso_x'].iloc[i] - df['torso_x'].iloc[i-1]
            dy = df['torso_y'].iloc[i] - df['torso_y'].iloc[i-1]
            dz = df['torso_z'].iloc[i] - df['torso_z'].iloc[i-1]
            torso_movement += np.sqrt(dx**2 + dy**2 + dz**2)
        metrics['torso_movement'] = torso_movement
    
    # Oscillation detection (high frequency changes in error)
    if len(df) > 10:
        error_deriv = df['error_norm'].diff().abs()
        metrics['oscillation_score'] = error_deriv.std()
    else:
        metrics['oscillation_score'] = 0
    
    return metrics

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_run(df, metrics, mode):
    """Generate plots for a run."""
    plt.style.use('seaborn')
    
    # Create output directory
    Path("logs/plots").mkdir(parents=True, exist_ok=True)
    
    # --- Plot 1: Target vs Actual Trajectory (XY view) ---
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(df['target_x'], df['target_y'], '--', linewidth=2, 
             color='red', label='Target Path')
    plt.plot(df['fl_x'], df['fl_y'], linewidth=2, label='FL Gripper')
    plt.plot(df['fr_x'], df['fr_y'], linewidth=1.5, label='FR Gripper', alpha=0.7)
    plt.plot(df['bl_x'], df['bl_y'], linewidth=1.5, label='BL Foot', alpha=0.7)
    plt.plot(df['br_x'], df['br_y'], linewidth=1.5, label='BR Foot', alpha=0.7)
    plt.scatter(df['fl_x'].iloc[0], df['fl_y'].iloc[0], color='green', s=50, label='Start')
    plt.scatter(df['fl_x'].iloc[-1], df['fl_y'].iloc[-1], color='blue', s=50, label='End')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'{mode} Mode - XY Trajectory\nError: {metrics["error_mean"]:.3f}m avg, {metrics["error_max"]:.3f}m max')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # --- Plot 2: Error over time ---
    plt.subplot(2, 2, 2)
    plt.plot(df['time'], df['error_norm'], linewidth=2, color='purple')
    plt.fill_between(df['time'], df['error_norm'], alpha=0.3, color='purple')
    plt.axhline(metrics['error_mean'], color='red', linestyle='--', 
                label=f'Mean: {metrics["error_mean"]:.3f}m')
    plt.xlabel('Time (s)')
    plt.ylabel('Tracking Error (m)')
    plt.title('FL Gripper Tracking Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # --- Plot 3: Control signals ---
    plt.subplot(2, 2, 3)
    control_cols = [c for c in df.columns if c.startswith('ctrl_')]
    if control_cols:
        for col in control_cols:
            plt.plot(df['time'], df[col], label=col, linewidth=1.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Control Signal')
        plt.title('Joint Control Signals')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # --- Plot 4: End effector positions over time ---
    plt.subplot(2, 2, 4)
    for ee in ['fl', 'fr', 'bl', 'br']:
        plt.plot(df['time'], df[f'{ee}_z'], label=f'{ee.upper()} Z', linewidth=1.5)
    plt.axhline(PATH_HEIGHT, color='gray', linestyle=':', label='Target Z')
    plt.xlabel('Time (s)')
    plt.ylabel('Z Position (m)')
    plt.title('End Effector Heights')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Agent Evaluation: {mode} Mode - {metrics["duration"]:.1f}s Run', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    
    # Save plots
    timestamp = Path(log_file).stem.split('_')[-2] if 'log_file' in globals() else 'latest'
    plot_path = f"logs/plots/eval_{mode}_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlots saved to: {plot_path}")
    
    plt.show()

# ============================================================================
# SCORE SYSTEM
# ============================================================================

def score_run(metrics, mode):
    """Generate a score (0-100) for the run quality."""
    score = 100
    reasons = []
    
    # Error penalty (most important)
    error_penalty = min(metrics['error_mean'] / 0.1, 1.0) * 40
    score -= error_penalty
    if error_penalty > 20:
        reasons.append(f"High tracking error: {metrics['error_mean']:.3f}m avg")
    
    # Oscillation penalty
    osc_penalty = min(metrics['oscillation_score'], 0.05) / 0.05 * 20
    score -= osc_penalty
    if osc_penalty > 10:
        reasons.append(f"Oscillations detected (score: {metrics['oscillation_score']:.4f})")
    
    # Path fidelity bonus
    if 'path_fidelity' in metrics:
        fidelity_bonus = (1 - metrics['path_fidelity']) * 20
        score -= fidelity_bonus
        if fidelity_bonus > 10:
            reasons.append(f"Path fidelity low: {metrics['path_fidelity']:.2%}")
    
    # Torso stability penalty
    if 'torso_movement' in metrics:
        torso_penalty = min(metrics['torso_movement'] / 0.1, 1.0) * 10
        score -= torso_penalty
        if torso_penalty > 5:
            reasons.append(f"Excessive torso movement: {metrics['torso_movement']:.3f}m")
    
    # Control effort penalty (too much effort = inefficient)
    if 'control_effort_mean' in metrics:
        effort_penalty = min(metrics['control_effort_mean'] / 5, 1.0) * 10
        score -= effort_penalty
    
    score = max(0, min(100, score))
    
    # Quality rating
    if score >= 80:
        rating = "EXCELLENT"
    elif score >= 60:
        rating = "GOOD"
    elif score >= 40:
        rating = "FAIR"
    elif score >= 20:
        rating = "POOR"
    else:
        rating = "FAIL"
    
    return score, rating, reasons

# ============================================================================
# COMPARISON
# ============================================================================

def compare_runs(files):
    """Compare multiple runs."""
    print(f"\n{'='*70}")
    print("COMPARING RUNS")
    print(f"{'='*70}\n")
    
    comparison = []
    for f in files:
        df = pd.read_csv(f)
        metrics = calculate_metrics(df)
        mode = Path(f).stem.split('_')[-1].upper()
        score, rating, reasons = score_run(metrics, mode)
        comparison.append({
            'file': Path(f).name,
            'mode': mode,
            'score': score,
            'rating': rating,
            'error_mean': metrics['error_mean'],
            'error_max': metrics['error_max'],
            'duration': metrics['duration'],
        })
    
    df_comp = pd.DataFrame(comparison)
    print(df_comp.to_string(index=False))
    print(f"\nBest run: {df_comp.loc[df_comp['score'].idxmax(), 'file']} ({df_comp['score'].max():.0f}/100)")
    
    return df_comp

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Parse arguments
    all_runs = '--all' in sys.argv
    mode_filter = None
    custom_pattern = None
    
    for arg in sys.argv[1:]:
        if arg == '--all':
            all_runs = True
        elif arg.startswith('--mode='):
            mode_filter = arg.split('=')[1]
        elif not arg.startswith('--'):
            custom_pattern = arg
    
    # Find log files
    if custom_pattern:
        files = find_logs(custom_pattern, all_runs, mode_filter)
    else:
        files = find_logs(all_runs=all_runs, mode_filter=mode_filter)
    
    if not files:
        sys.exit(1)
    
    # Process each file
    for log_file_path in files:
        log_file = Path(log_file_path)
        print(f"\n{'='*70}")
        print(f"Evaluating: {log_file.name}")
        print(f"{'='*70}")
        
        df = pd.read_csv(log_file)
        mode = log_file.stem.split('_')[-1].upper()
        metrics = calculate_metrics(df)
        score, rating, reasons = score_run(metrics, mode)
        
        # Print metrics
        print(f"\nMode: {mode}")
        print(f"Rating: {rating} ({score:.0f}/100)")
        print(f"\nDuration: {metrics['duration']:.2f}s")
        print(f"Tracking Error: {metrics['error_mean']:.4f}m (max: {metrics['error_max']:.4f}m)")
        
        if 'path_fidelity' in metrics:
            print(f"Path Fidelity: {metrics['path_fidelity']:.1%}")
        
        if 'torso_movement' in metrics:
            print(f"Torso Movement: {metrics['torso_movement']:.4f}m")
        
        if reasons:
            print(f"\nIssues detected:")
            for reason in reasons:
                print(f"  - {reason}")
        else:
            print(f"\nNo major issues detected!")
        
        # Generate plots
        plot_run(df, metrics, mode)
    
    # Compare if multiple files
    if len(files) > 1 and all_runs:
        compare_runs(files)
