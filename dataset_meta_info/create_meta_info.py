import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
from tqdm import tqdm
import torch
import random
import imageio
from decord import VideoReader, cpu
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
# from finetune.constants import LOG_LEVEL, LOG_NAME
import numpy as np
from scipy.spatial.transform import Rotation as R  


def extract_state_matrix(ann: Dict[str, Any]):
    if 'states' in ann and len(ann['states']) > 0:
        return np.array(ann['states'])

    if 'observation.state.cartesian_position' in ann and 'observation.state.gripper_position' in ann:
        cartesian = np.array(ann['observation.state.cartesian_position'])
        gripper = np.array(ann['observation.state.gripper_position'])
        if gripper.ndim == 1:
            gripper = gripper[..., np.newaxis]
        return np.concatenate([cartesian, gripper], axis=-1)

    required_keys = [
        'observation.state.joint_position_arm_left',
        'observation.state.gripper_position_left',
        'observation.state.joint_position_arm_right',
        'observation.state.gripper_position_right',
        'observation.state.joint_position_torso',
    ]
    if all(key in ann for key in required_keys):
        arm_left = np.array(ann['observation.state.joint_position_arm_left'])
        grip_left = np.array(ann['observation.state.gripper_position_left'])
        arm_right = np.array(ann['observation.state.joint_position_arm_right'])
        grip_right = np.array(ann['observation.state.gripper_position_right'])
        torso = np.array(ann['observation.state.joint_position_torso'])

        if grip_left.ndim == 1:
            grip_left = grip_left[..., np.newaxis]
        if grip_right.ndim == 1:
            grip_right = grip_right[..., np.newaxis]

        return np.concatenate([arm_left, grip_left, arm_right, grip_right, torso], axis=-1)

    return None

def load_and_process_ann_file(data_root, ann_file, sequence_interval=1, start_interval=4, sequence_length=8):
    samples = []
    try:
        with open(f'{data_root}/{ann_file}', "r") as f:
            ann = json.load(f)
    except Exception as exc:
        print(f'skip {ann_file}: {exc}')
        return samples

    state_matrix = extract_state_matrix(ann)
    if state_matrix is None or state_matrix.shape[0] == 0:
        print(f'skip {ann_file}: unsupported annotation format')
        return samples

    n_frames = min(ann.get('video_length', state_matrix.shape[0]), state_matrix.shape[0])
    traj_len = int(sequence_length*sequence_interval)
    end_idx = n_frames - int(traj_len*0.5)
    if end_idx < 1:
        end_idx = 1

    for start_frame in range(0,end_idx,start_interval):       
        idx = start_frame
        sample = dict()
        sample['episode_id'] = ann['episode_id']
        sample['frame_ids'] = [idx]
        sample['states'] = state_matrix[idx:idx+1]
        samples.append(sample)
    return samples

def init_anns(dataset_root, data_dir):
    final_path = f'{dataset_root}/{data_dir}'
    ann_files = [os.path.join(data_dir, f) for f in os.listdir(final_path) if f.endswith('.json')]
    return ann_files

def init_sequences(data_root, ann_files, sequence_interval, start_interval,sequence_length):
    samples = []
    with ThreadPoolExecutor(32) as executor:
        future_to_ann_file = {executor.submit(load_and_process_ann_file, data_root, ann_file, sequence_interval, start_interval, sequence_length): ann_file for ann_file in ann_files}
        for future in tqdm(as_completed(future_to_ann_file), total=len(ann_files)):
            samples.extend(future.result())
    return samples


if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--droid_output_path', type=str, default='dataset_example/droid_subset')
    # dataset_name
    parser.add_argument('--dataset_name', type=str, default='droid_subset')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    ########################### dataset statistics ###########################
    sequence_length = 8
    combined_states = []
    for data_type in ['val', 'train']:
        samples_all = []
        ann_files_all = []
        data_root = args.droid_output_path
        dataset_name = args.dataset_name

        sequence_interval = 1
        start_interval = 1
        ann_dir = f'annotation/{data_type}'
        ann_files = init_anns(data_root, ann_dir)
        ann_files_all.extend(ann_files)
        samples = init_sequences(data_root, ann_files, sequence_interval, start_interval, sequence_length)
        print(f'{data_root} {len(samples)} samples')
        samples_all.extend(samples)

        # accumulate state statistics
        state_vectors = []
        for sample in samples:
            state_arr = np.array(sample['states'])
            state_arr = state_arr.reshape(-1, state_arr.shape[-1])
            state_vectors.append(state_arr)
        if len(state_vectors) > 0:
            combined_states.append(np.concatenate(state_vectors, axis=0))
        
        # dataset meta info
        for sample in samples_all:
            if 'states' in sample:
                del sample['states']
        import random
        random.shuffle(samples_all)
        print('step_num',data_type,len(samples_all))
        print('traj_num',data_type, len(ann_files_all))
        os.makedirs(f'dataset_meta_info/{dataset_name}', exist_ok=True)
        with open(f'dataset_meta_info/{dataset_name}/{data_type}_sample.json', 'w') as f:
            json.dump(samples_all, f, indent=4)

    if len(combined_states) > 0:
        state_all = np.concatenate(combined_states, axis=0)
        state_01 = np.percentile(state_all, 1, axis=0)
        state_99 = np.percentile(state_all, 99, axis=0)
        stat = {
            'state_01': state_01.tolist(),
            'state_99': state_99.tolist(),
        }
        os.makedirs(f'dataset_meta_info/{args.dataset_name}', exist_ok=True)
        with open(f'dataset_meta_info/{args.dataset_name}/stat.json', 'w') as f:
            json.dump(stat, f, indent=4)
        print('Saved normalization statistics to', f'dataset_meta_info/{args.dataset_name}/stat.json')
    else:
        print('Warning: no state statistics were collected; stat.json will not be generated.')
        
