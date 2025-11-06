import json
import os
import random
import warnings
import traceback
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms as T
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import imageio
from decord import VideoReader, cpu
from concurrent.futures import ThreadPoolExecutor, as_completed
from einops import rearrange
from scipy.spatial.transform import Rotation as R  
import decord


def _extract_state_matrix(label):
    """Extracts the state matrix from a trajectory label across supported datasets."""
    if 'states' in label and len(label['states']) > 0:
        return np.array(label['states'])

    if 'observation.state.cartesian_position' in label and 'observation.state.gripper_position' in label:
        cartesian = np.array(label['observation.state.cartesian_position'])
        gripper = np.array(label['observation.state.gripper_position'])
        if gripper.ndim == 1:
            gripper = gripper[..., np.newaxis]
        return np.concatenate([cartesian, gripper], axis=-1)

    # Galaxea-style dataset
    required_keys = [
        'observation.state.joint_position_arm_left',
        'observation.state.gripper_position_left',
        'observation.state.joint_position_arm_right',
        'observation.state.gripper_position_right',
        'observation.state.joint_position_torso',
    ]
    if all(key in label for key in required_keys):
        arm_left = np.array(label['observation.state.joint_position_arm_left'])
        grip_left = np.array(label['observation.state.gripper_position_left'])
        arm_right = np.array(label['observation.state.joint_position_arm_right'])
        grip_right = np.array(label['observation.state.gripper_position_right'])
        torso = np.array(label['observation.state.joint_position_torso'])

        if grip_left.ndim == 1:
            grip_left = grip_left[..., np.newaxis]
        if grip_right.ndim == 1:
            grip_right = grip_right[..., np.newaxis]

        return np.concatenate(
            [arm_left, grip_left, arm_right, grip_right, torso], axis=-1
        )

    return None

class Dataset_mix(Dataset):
    def __init__(
            self,
            args,
            mode = 'val',
    ):
        """Constructor."""
        super().__init__()
        self.args = args
        self.mode = mode

        # dataset stucture
        # dataset_root_path/dataset_name/annotation_name/mode/traj
        # dataset_root_path/dataset_name/video/mode/traj
        # dataset_root_path/dataset_name/latent_video/mode/traj

        # samples:{'ann_file':xxx, 'frame_idx':xxx, 'dataset_name':xxx}

        # prepare all datasets path
        self.dataset_path_all = []
        self.samples_all = []
        self.samples_len = []
        self.norm_all = []


        dataset_root_path = args.dataset_root_path
        dataset_names = args.dataset_names.split('+')
        dataset_meta_info_path = args.dataset_meta_info_path
        dataset_cfgs = args.dataset_cfgs.split('+')
        self.prob = args.prob
        for dataset_name, dataset_cfg in zip(dataset_names, dataset_cfgs):
            data_json_path = f'{dataset_meta_info_path}/{dataset_cfg}/{mode}_sample.json'
     
            with open(data_json_path, "r") as f:
                samples = json.load(f)
            dataset_path = [os.path.join(dataset_root_path, dataset_name) for sample in samples]
            print(f"ALL dataset, {len(samples)} samples in total")
            self.dataset_path_all.append(dataset_path)
            self.samples_all.append(samples)
            self.samples_len.append(len(samples))

            # prepare normalization
            with open(f'{dataset_meta_info_path}/{dataset_name}/stat.json', "r") as f:
                data_stat = json.load(f)
                state_p01 = np.array(data_stat['state_01'])[None,:]
                state_p99 = np.array(data_stat['state_99'])[None,:]
                self.norm_all.append((state_p01, state_p99))
        
        self.max_id = max(self.samples_len)
        print('samples_len:',self.samples_len, 'max_id:',self.max_id)

    def __len__(self):
        return self.max_id

    def _resolve_latent_entries(self, label):
        entries = []
        if 'latent_videos' in label and label['latent_videos']:
            for item in label['latent_videos']:
                if isinstance(item, dict):
                    entries.append(item.get('latent_video_path'))
                else:
                    entries.append(item)
        elif 'videos' in label and label['videos']:
            for item in label['videos']:
                if isinstance(item, dict) and 'latent_video_path' in item:
                    entries.append(item['latent_video_path'])
                else:
                    raise ValueError("Missing latent video path for one of the camera views. Please precompute latents for the dataset.")
        else:
            raise ValueError("Annotation file does not contain latent video information.")

        if any(entry is None for entry in entries):
            raise ValueError("Incomplete latent video specification detected in annotation file.")
        return entries

    def _infer_state_length(self, label, state_matrix):
        if state_matrix is not None:
            return state_matrix.shape[0]
        if 'observation.state.joint_position' in label:
            return len(label['observation.state.joint_position'])
        if 'observation.state.joint_position_arm_left' in label:
            return len(label['observation.state.joint_position_arm_left'])
        if 'states' in label:
            return len(label['states'])
        raise ValueError("Unable to infer trajectory length from annotation.")

    def _load_latent_video(self, video_path, frame_ids):
        with open(video_path,'rb') as file:
            video_tensor = torch.load(file)
            video_tensor.requires_grad = False
        max_frames = video_tensor.size()[0]
        frame_ids =  [int(frame_id) if frame_id < max_frames else max_frames-1 for frame_id in frame_ids]
        frame_data = video_tensor[frame_ids]
        return frame_data

    def _get_frames(self, label, frame_ids, cam_id, pre_encode, video_dir, use_img_cond=False, latent_entries=None):
        # directly load videos latent after svd-vae encoder
        assert cam_id is not None
        assert pre_encode == True
        if pre_encode:
            entries = latent_entries if latent_entries is not None else self._resolve_latent_entries(label)
            if cam_id >= len(entries):
                raise IndexError(f"Camera id {cam_id} is out of range for available latent videos ({len(entries)} views).")
            video_path = entries[cam_id]
            video_path = os.path.join(video_dir, video_path)
            try:
                frames = self._load_latent_video(video_path, frame_ids)
            except FileNotFoundError:
                video_path = video_path.replace("latent_videos", "latent_videos_svd")
                frames = self._load_latent_video(video_path, frame_ids)
            except Exception as exc:
                raise RuntimeError(f"Failed to load latent video from {video_path}: {exc}") from exc
        return frames

    def _get_obs(self, label, frame_ids, cam_id, pre_encode, video_dir, latent_entries=None):
        if cam_id is None:
            temp_cam_id = random.choice(range(len(self._resolve_latent_entries(label))))
        else:
            temp_cam_id = cam_id
        frames = self._get_frames(label, frame_ids, cam_id=temp_cam_id, pre_encode=pre_encode, video_dir=video_dir, latent_entries=latent_entries)
        return frames, temp_cam_id

    def normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def denormalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps=1e-8,
    ) -> np.ndarray:
        clip_range = clip_max - clip_min
        rdata = (data - clip_min) / clip_range * (data_max - data_min) + data_min
        return rdata

    def __getitem__(self, index):

        # first sample the dataset id, than sample the data from the dataset
        dataset_id = np.random.choice(len(self.samples_all), p=self.prob)
        samples = self.samples_all[dataset_id]
        dataset_path = self.dataset_path_all[dataset_id]
        state_p01, state_p99 = self.norm_all[dataset_id]
        index = index % len(samples)
        sample = samples[index]
        dataset_dir = dataset_path[index]

        # get annotation
        frame_ids = sample['frame_ids']
        ann_file = f'{dataset_dir}/{self.args.annotation_name}/{self.mode}/{sample["episode_id"]}.json'
        with open(ann_file, "r") as f:
            label = json.load(f)
            
        state_matrix = _extract_state_matrix(label)
        if state_matrix is None:
            raise ValueError(f"Unsupported annotation format for episode {sample['episode_id']} in dataset {dataset_dir}.")

        traj_len = self._infer_state_length(label, state_matrix)
        joint_len = max(traj_len - 1, 0)
        frame_scale = max(self.args.down_sample, 1)
        if 'video_length' in label and label['video_length'] is not None:
            frame_len = max(int(label['video_length']) - 1, 0)
        else:
            frame_len = int(np.floor(joint_len / frame_scale))
        skip = random.randint(1, 2)
        skip_his = int(skip*4)
        p = random.random()
        if p < 0.15:
            skip_his = 0
        
        # rgb_id and state_id
        frame_now = frame_ids[0]
        rgb_id = []
        for i in range(self.args.num_history,0,-1):
            rgb_id.append(int(frame_now - i*skip_his))
        rgb_id.append(frame_now)
        for i in range(1, self.args.num_frames):
            rgb_id.append(int(frame_now + i*skip))
        rgb_id = np.array(rgb_id)
        rgb_id = np.clip(rgb_id, 0, frame_len).astype(int).tolist()
        state_id = np.array(rgb_id) * frame_scale
        state_id = np.clip(state_id, 0, traj_len - 1).astype(int)


        # prepare data
        data = dict()

        # instructions
        data['text'] = label.get('texts', [""])[0]

        # stack tokens of multi-view
        latent_entries = self._resolve_latent_entries(label)
        if len(latent_entries) == 0:
            raise ValueError(f"No latent videos found for episode {sample['episode_id']}.")

        cam_latents = []
        for cam_id in range(len(latent_entries)):
            latnt_cond, _ = self._get_obs(label, rgb_id, cam_id, pre_encode=True, video_dir=dataset_dir, latent_entries=latent_entries)
            cam_latents.append(latnt_cond)

        try:
            latent = torch.cat(cam_latents, dim=2)
        except RuntimeError as exc:
            raise RuntimeError(f"Failed to concatenate latent videos for episode {sample['episode_id']}: {exc}") from exc
        data['latent'] = latent.float()

        # prepare action cond data
        action = state_matrix[state_id]
        action = self.normalize_bound(action, state_p01, state_p99)
        data['action'] = torch.tensor(action).float()
        data['ann_file'] = ann_file

        return data
        

if __name__ == "__main__":

    from config import wm_args
    args = wm_args()
    train_dataset = Dataset_mix(args,mode="val")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True
    )
    for data in tqdm(train_loader,total=len(train_loader)):
        print(data['ann_file'])

    