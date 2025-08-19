
import json
import pathlib
import yt_dlp               # for downloading YouTube videos
import ffmpeg             # for trimming videos using ffmpeg
import tqdm               #for showing progress bars
import os

# Add ffmpeg to system PATH so it can be called from code
os.environ["PATH"] += os.pathsep + r"C:\installed\programs\ffmpeg-7.1.1-full_build\ffmpeg-7.1.1-full_build\bin"

# Set up base directories
base_path = pathlib.Path("C:/Users/seman/Desktop/clg/2nd_sem/research_practicum/code")
raw_root = base_path / "data" / "raw"        # Where full YouTube videos will be stored
clips_root = base_path / "data" / "clips"    # Where trimmed clips will be saved

# Download a YouTube video to the raw folder
def download_youtube_video(url, raw_path):
    if raw_path.exists():
        return  # Skip download if already exists

    try:
        yt_dlp.YoutubeDL({
            'outtmpl': str(raw_path),                          # Output file path
            'format': 'bestvideo[ext=mp4]',                    # Best MP4 video format
            'ffmpeg_location': r'C:\installed\programs\ffmpeg-7.1.1-full_build\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe',
            'quiet': True                                      # Suppress yt-dlp console output
        }).download([url])
    except Exception as e:
        print(f"[ERROR] Failed to download: {url}\nReason: {e}")

# Trim the video using ffmpeg from start_time to end_time
def trim_video(raw_path, clip_path, start_time, end_time):
    if clip_path.exists():
        return  # Skip if already trimmed

    try:
        (
            ffmpeg
            .input(str(raw_path), ss=start_time, to=end_time)     # Input with start and end time
            .output(str(clip_path), codec='copy', loglevel='quiet')  # Copy codec, no re-encoding
            .run()                                                # Run the ffmpeg command
        )
    except Exception as e:
        print(f"[ERROR] Failed to trim: {clip_path}\nReason: {e}")

# Process all clips for a given split (train/val/test)
def process_split(split_name):
    print(f"\n[PROCESSING SPLIT] {split_name.upper()}")

    # Path to JSON file containing clip info
    json_path = base_path / "data" / "lists" / f"ASL100_{split_name}.json"
    raw_split_root = raw_root / split_name              # Raw video path for this split
    clip_split_root = clips_root / split_name           # Trimmed clips path for this split

    # Create folders if they don't exist
    raw_split_root.mkdir(parents=True, exist_ok=True)
    clip_split_root.mkdir(parents=True, exist_ok=True)

    # Load clip metadata from JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        clip_list = json.load(f)

    # Process each clip in the JSON list
    for clip in tqdm.tqdm(clip_list, desc=f"{split_name}"):
        class_name = clip["clean_text"]      # Name of the ASL class/label

        # Prepare folders for this class
        class_raw_dir = raw_split_root / class_name
        class_clip_dir = clip_split_root / class_name
        class_raw_dir.mkdir(parents=True, exist_ok=True)
        class_clip_dir.mkdir(parents=True, exist_ok=True)

        # Extract YouTube video ID
        video_id = clip["url"].split("v=")[-1]

        # Define raw and trimmed video file paths
        raw_file = class_raw_dir / f"{video_id}.mp4"
        trimmed_filename = clip["file"].replace(" ", "_") + ".mp4"  # Sanitize filename
        trimmed_file = class_clip_dir / trimmed_filename

        # Step 1: Download full YouTube video
        download_youtube_video(clip["url"], raw_file)

        # Step 2: Trim the clip from the full video
        trim_video(raw_file, trimmed_file, clip["start_time"], clip["end_time"])

# Main script: run for all splits (train, val, test)
if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        process_split(split)
