import gc
import os

import torch
from ..tools import *

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = torch.device("cuda")
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

from ..sam2.build_sam import build_sam2_video_predictor
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)


def propagate(opt, videos_path, propagations_path):
    """
    Propagate each frame within the temporal window using SAM2
    """
    frames_list = sorted(os.listdir(opt.frames_path))
    mask_type = os.listdir(opt.origin_masks_path)[0].split('.')[-1]
    for idx, frame_name in enumerate(frames_list):
        prompt_path = os.path.join(opt.origin_masks_path, f"{frame_name[:-4]}.{mask_type}")
        ids, prompt_boxes, center_points = extract_bboxes(prompt_path)
        if ids == 1:  # negative frame
            continue

        video_dir = os.path.join(videos_path, "{:04d}".format(idx))
        video_frames = sorted(os.listdir(video_dir))
        prompt_index = video_frames.index(frame_name)

        inference_state = predictor.init_state(video_path=video_dir)

        reverses = []
        if prompt_index != 0:
            reverses.append(True)
        if prompt_index != len(video_frames) - 1:
            reverses.append(False)

        for id in range(1, ids):
            points = np.array([center_points[id - 1]], dtype=np.float32)
            labels = np.array([1], np.int32)
            box = np.array(prompt_boxes[id - 1], dtype=np.float32)
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=prompt_index,
                obj_id=id,
                points=points,
                labels=labels,
                box=box,
            )

        for reverse in reverses:
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,
                                                                                            reverse=reverse):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            for i in range(len(video_segments)):
                if not reverse:
                    test_index = prompt_index + i
                    receive_idx = idx + i
                else:
                    test_index = prompt_index - i
                    receive_idx = idx - i
                file_name = "{:04d}".format(idx) + "-" + "{:04d}".format(receive_idx) + ".jpg"
                masks = video_segments[test_index]
                res = np.zeros_like(masks[1][0], np.uint8)
                for id, mask in masks.items():
                    mask = mask[0].astype(np.uint8) * 255
                    res += mask
                cv2.imwrite(os.path.join(propagations_path, file_name), res)

        del inference_state
        gc.collect()
        torch.cuda.empty_cache()


def calc_SQA(opt, propagations_path):
    """
    calculate the Segmentation Quality Assessment (SQA) Scores
    """
    propagations_list = os.listdir(propagations_path)
    prop_dice = {}
    self_dice = {}
    SQA_dict = {}
    for name in os.listdir(opt.frames_path):
        key = name.split('.')[0]
        prop_dice[key] = {}
    mask_type = os.listdir(opt.origin_masks_path)[0].split('.')[-1]

    for prop_name in propagations_list:
        prompt_name = prop_name[:-4].split("-")[0]
        receive_name = prop_name[:-4].split("-")[1]

        dis = int(receive_name) - int(prompt_name)
        if abs(dis) > opt.radius:
            continue

        ori_mask = os.path.join(opt.origin_masks_path, f"{receive_name}.{mask_type}")
        prop_mask = os.path.join(propagations_path, prop_name)
        dice = Dice(ori_mask, prop_mask)

        if dis == 0:
            self_dice[receive_name] = dice
        else:
            prop_dice[receive_name][dis] = dice

    assert len(prop_dice) == len(os.listdir(opt.frames_path))
    if len(self_dice) < len(prop_dice):
        for key, _ in prop_dice.items():
            if key not in self_dice.keys():
                self_dice[key] = 0

    sorted_self_dict = {k: self_dice[k] for k in sorted(self_dice)}
    sorted_prop_dict = {k: prop_dice[k] for k in sorted(prop_dice)}

    for key, dices in sorted_prop_dict.items():
        if len(dices) != 0:
            avg = sum(dices.values()) / len(dices.keys())
        else:
            avg = 0
        SQA_dict[key] = avg

    with open("./temp/self_dice.json", 'w') as file1:
        json.dump(sorted_self_dict, file1)
    with open(opt.SQA_scores_path, 'w') as file3:
        json.dump(SQA_dict, file3)
