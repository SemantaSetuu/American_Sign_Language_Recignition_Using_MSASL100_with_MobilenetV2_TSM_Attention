#extract_frames.py
import cv2
from pathlib import Path

def extract_frames_from_videos(input_dir, output_dir, fps=16):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]

    for class_dir in input_dir.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            for video_file in class_dir.iterdir():
                if video_file.suffix.lower() in video_extensions:
                    cap = cv2.VideoCapture(str(video_file))
                    video_name = video_file.stem  # e.g. again_sign01

                    # 👇 Create subfolder: output/class_name/video_name/
                    save_path = output_dir / class_name / video_name
                    save_path.mkdir(parents=True, exist_ok=True)

                    frame_count = 0
                    frame_index = 0
                    frame_rate = cap.get(cv2.CAP_PROP_FPS)
                    frame_interval = int(frame_rate // fps) if frame_rate >= fps else 1

                    success, frame = cap.read()
                    while success:
                        if frame_count % frame_interval == 0:
                            frame_filename = save_path / f"{video_name}_{frame_index}.jpg"
                            cv2.imwrite(str(frame_filename), frame)
                            frame_index += 1
                        success, frame = cap.read()
                        frame_count += 1

                    cap.release()
                    print(f" Extracted frames from {video_file}")

def main():
    sets = ["train", "val", "test"]
    for s in sets:
        video_root = fr"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code\data\clips\{s}"
        image_root = fr"C:\Users\seman\Desktop\clg\2nd_sem\research_practicum\code\data\images\{s}"
        extract_frames_from_videos(video_root, image_root, fps=16)

if __name__ == "__main__":
    main()
