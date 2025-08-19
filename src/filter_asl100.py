import json
from pathlib import Path

# Define the root directory of the project where "code" folder lives
base_path = Path("C:/Users/seman/Desktop/clg/2nd_sem/research_practicum/code")

def main(msasl_root: Path):
    # Step 1: Load the full list of all sign classes (1000+ classes)
    msasl_class_file = msasl_root / "MS-ASL" / "MSASL_classes.json"
    with open(msasl_class_file, 'r', encoding='utf-8') as f:
        all_classes = json.load(f)

    # Step 2: Select only the first 100 class names
    first_100_signs = all_classes[:100]
    keep_set = set(first_100_signs)  # convert to set for faster lookup

    # Step 3: Loop through all dataset splits
    for split_name in ['train', 'val', 'test']:
        # Step 3.1: Load the full split file (train/val/test)
        split_file = msasl_root / "MS-ASL" / f"MSASL_{split_name}.json"
        with open(split_file, 'r', encoding='utf-8') as infile:
            video_list = json.load(infile)

        # Step 3.2: Filter videos that belong to the selected 100 signs
        filtered_list = []
        for video_data in video_list:
            if video_data['clean_text'] in keep_set:
                filtered_list.append(video_data)

        # Step 3.3: Define path to save the new filtered JSON file
        output_path = msasl_root / "data" / "lists" / f"ASL100_{split_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Step 3.4: Save the filtered list to the new file
        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(filtered_list, outfile, indent=2)

        print(f"{output_path.name}: {len(filtered_list)} clips")

if __name__ == '__main__':
    main(base_path)
