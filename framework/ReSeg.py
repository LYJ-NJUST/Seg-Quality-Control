import torch
from ..tools import *

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = torch.device("cuda")
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

from ..sam2.build_sam import build_sam2_video_predictor
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)


def remove_small_regions(mask: np.ndarray, area_thresh: float):
    """
    Removes small disconnected regions and holes in a mask.
    """
    num_labels0, labels0, stats0, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels0):
        if stats0[i, cv2.CC_STAT_AREA] < area_thresh:
            labels0[labels0 == i] = 0
    res0 = (labels0 == 0).astype(np.uint8) * 255

    num_labels1, labels1, stats1, _ = cv2.connectedComponentsWithStats(res0, connectivity=8)
    for i in range(1, num_labels1):
        if stats1[i, cv2.CC_STAT_AREA] < area_thresh:
            labels1[labels1 == i] = 0
    res1 = (labels1 == 0).astype(np.uint8) * 255

    return res1


def select_highqua(SQA_scores, self_scores, avg_score, thresh):
    highquas = []
    for i, (S, s) in enumerate(zip(SQA_scores, self_scores)):
        if S >= avg_score and s >= thresh:
            highquas.append(i)
    return highquas


def find_prompt(idx, highquas, num):
    distance_pairs = [(highqua, abs(highqua - idx)) for highqua in highquas]
    distance_pairs.sort(key=lambda x: x[1])
    prompt_list = [pair[0] for pair in distance_pairs[:num]]
    return prompt_list


def SAM2_reseg(frames_path, masks_path, idx, prompt_idx):
    create_video(frames_path, min(idx, prompt_idx), max(idx, prompt_idx), "./temp/reseg_video")
    prompt_name = "{:04d}".format(prompt_idx)
    mask_type = os.listdir(masks_path)[0].split('.')[-1]
    prompt_path = os.path.join(masks_path, f"{prompt_name}.{mask_type}")
    ids, prompt_boxes, center_points = extract_bboxes(prompt_path)

    if prompt_idx < idx:
        index = 0
        reverse = False
    else:
        index = prompt_idx - idx
        reverse = True
    inference_state = predictor.init_state(video_path="./temp/reseg_video")

    for id in range(1, ids):
        points = np.array([center_points[id - 1]], dtype=np.float32)
        labels = np.array([1], np.int32)
        box = np.array(prompt_boxes[id - 1], dtype=np.float32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=index,
            obj_id=id,
            points=points,
            labels=labels,
            box=box,
        )

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,
                                                                                    reverse=reverse):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    if not reverse:
        test_index = idx - prompt_idx
    else:
        test_index = 0

    masks = video_segments[test_index]
    res = np.zeros_like(masks[1][0], np.uint8)
    for id, mask in masks.items():
        mask = mask[0].astype(np.uint8) * 255
        res += mask

    destroy_dirs(["./temp/reseg_video"])
    del inference_state
    return res


def ReSeg(opt, propagations_path):
    with open(opt.SQA_scores_path, 'r') as file1:
        score_dict = json.load(file1)
    with open("./temp/self_dice.json", 'r') as file2:
        self_dict = json.load(file2)

    mask_type = os.listdir(opt.origin_masks_path)[0].split('.')[-1]
    SQA_scores = [score_dict[k] for k in sorted(score_dict)]
    self_scores = [self_dict[k] for k in sorted(self_dict)]
    assert len(SQA_scores) == len(self_scores)

    avg_score = sum(SQA_scores) / len(SQA_scores)
    highquas = select_highqua(SQA_scores, self_scores, avg_score, opt.threshold)  # select high quality frames

    for idx, score in enumerate(SQA_scores):
        key = "{:04d}".format(idx)
        if score < avg_score:  # low quality frame -- reseg
            prompt_list = find_prompt(idx, highquas, opt.reseg_num)
            res = np.zeros_like(cv2.imread(os.path.join(opt.origin_masks_path, f"{key}.{mask_type}"), 0))
            for prompt_idx in prompt_list:
                prompt_key = "{:04d}".format(prompt_idx)
                if abs(idx - prompt_idx) <= opt.radius:
                    mask_name = f"{prompt_key}-{key}.jpg"
                    mask = cv2.imread(os.path.join(propagations_path, mask_name), 0)
                else:
                    mask = SAM2_reseg(opt.frames_path, opt.origin_masks_path, idx, prompt_idx)
                res += mask
            res = torch.from_numpy(res).sigmoid().data.cpu().numpy().squeeze()
            res[res >= 1] = 255
            res[res < 1] = 0
            res = res.astype(np.uint8)
            res = remove_small_regions(res, 200)
            cv2.imwrite(os.path.join(opt.ReSeg_masks_path, f"{key}.jpg"), res)
        else:  # high quality frame -- remain unchanged
            shutil.copy(os.path.join(opt.origin_masks_path, f"{key}.jpg"), os.path.join(opt.ReSeg_masks_path, f"{key}.jpg"))
