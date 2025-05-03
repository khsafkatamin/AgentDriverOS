import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../agentdriver')))

import json
import random
from pathlib import Path

from unittest.mock import MagicMock

# 👇 Mock tensorflow before importing anything that depends on it
sys.modules['openai'] = MagicMock()
sys.modules['tenacity'] = MagicMock()
sys.modules['skimage'] = MagicMock()
sys.modules['skimage.draw'] = MagicMock()
sys.modules['shapely'] = MagicMock()
sys.modules['shapely.geometry'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['casadi'] = MagicMock()
sys.modules['cv2'] = MagicMock()

from agentdriver.planning.planning_prmopts import planning_system_message as system_message
from agentdriver.planning.motion_planning import generate_messages


def generate_hf_finetune_data(data_path, data_file, sample_ratio=1.0, use_gt_cot=False):
    """
    Generate Hugging Face SFT-compatible format from planning data.
    Each example will be a dict: {"prompt": <input>, "response": <output>}
    """

    # Load and sample
    data_samples = json.load(open(Path(data_path) / Path(data_file), 'r'))
    sample_size = int(len(data_samples) * sample_ratio)
    data_samples = random.sample(data_samples, sample_size)

    hf_train_data = []
    invalid_count = 0

    for data_sample in data_samples:
        token, user_message, assistant_message = generate_messages(data_sample, use_gt_cot=use_gt_cot, verbose=False)
        if not assistant_message:
            continue

        # Format for Hugging Face fine-tuning
        formatted_input = f"<s>[SYSTEM] {system_message} [/SYSTEM]\n[USER] {user_message} [/USER]\n[ASSISTANT]"
        formatted_output = assistant_message.strip()

        hf_train_data.append({
            "prompt": formatted_input,
            "response": formatted_output
        })

    print("#### Hugging Face Data Summary ####")
    print(f"Number of total training pairs: {len(hf_train_data)}")

    saved_file_name = "hf_finetune_planner_" + str(int(sample_ratio * 100)) + ".json"
    with open(Path(data_path) / Path(saved_file_name), "w") as f:
        json.dump(hf_train_data, f, indent=2, ensure_ascii=False)

    print(f"Saved Hugging Face fine-tune file to: {saved_file_name}")


if __name__ == "__main__":
    generate_hf_finetune_data(
        data_path="data/finetune",
        data_file="data_samples_train.json",
        sample_ratio=1.0,
        use_gt_cot=False
    )