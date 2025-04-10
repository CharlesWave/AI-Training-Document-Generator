#!/usr/bin/env python3
"""
Video Splitter

This script splits a one-hour MP4 video into smaller 10-minute segments with 2-minute overlaps
between consecutive segments.

Requirements:
- Python 3.6+
- FFmpeg installed and available in PATH
- ffmpeg-python package

Usage:
    python video_splitter.py input_video.mp4 output_directory
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime, timedelta

def time_to_seconds(time_str):
    """Convert time string in format HH:MM:SS to seconds."""
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def seconds_to_time(seconds):
    """Convert seconds to time string in format HH:MM:SS."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def get_video_duration(video_path):
    """Get the duration of a video file in seconds using ffprobe."""
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        video_path
    ]
    
    try:
        output = subprocess.check_output(cmd).decode('utf-8').strip()
        return float(output)
    except subprocess.CalledProcessError as e:
        print(f"Error getting video duration: {e}")
        sys.exit(1)

def split_video(input_video, output_dir, segment_length=600, overlap=120):
    """
    Split a video into segments with overlap.
    
    Args:
        input_video: Path to the input video file
        output_dir: Directory to save the output segments
        segment_length: Length of each segment in seconds (default: 10 minutes)
        overlap: Overlap between segments in seconds (default: 2 minutes)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video duration
    duration = get_video_duration(input_video)
    print(f"Video duration: {seconds_to_time(int(duration))}")
    
    # Calculate start times for each segment
    start_times = []
    current_time = 0
    
    while current_time < duration:
        start_times.append(current_time)
        current_time += segment_length - overlap
    
    # Process each segment
    for i, start_time in enumerate(start_times):
        # Calculate end time (either segment_length or end of video)
        end_time = min(start_time + segment_length, duration)
        
        # Skip if segment would be too short (less than 10 seconds)
        if end_time - start_time < 10:
            continue
        
        # Format times for ffmpeg
        start_time_str = seconds_to_time(int(start_time))
        segment_duration = end_time - start_time
        
        # Output filename
        output_file = os.path.join(
            output_dir, 
            f"segment_{i+1:02d}_{start_time_str.replace(':', '')}_to_{seconds_to_time(int(end_time)).replace(':', '')}.mp4"
        )
        
        # FFmpeg command
        cmd = [
            "ffmpeg",
            "-i", input_video,
            "-ss", start_time_str,
            "-t", str(segment_duration),
            "-c", "copy",  # Copy streams without re-encoding
            "-avoid_negative_ts", "1",
            output_file
        ]
        
        print(f"Creating segment {i+1}: {start_time_str} to {seconds_to_time(int(end_time))}")
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully created {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error creating segment: {e}")

def main():
    parser = argparse.ArgumentParser(description="Split a video into segments with overlap")
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument("output_dir", help="Directory to save the output segments")
    parser.add_argument("--segment-length", type=int, default=600,
                        help="Length of each segment in seconds (default: 600 seconds = 10 minutes)")
    parser.add_argument("--overlap", type=int, default=120,
                        help="Overlap between segments in seconds (default: 120 seconds = 2 minutes)")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.isfile(args.input_video):
        print(f"Error: Input file '{args.input_video}' does not exist")
        sys.exit(1)
    
    # Split the video
    split_video(args.input_video, args.output_dir, args.segment_length, args.overlap)
    
    print("Video splitting completed!")

if __name__ == "__main__":
    main() 