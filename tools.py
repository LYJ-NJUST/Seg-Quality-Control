import os
import cv2
import json
import shutil
import numpy as np
from scipy import stats


def make_dirs(paths: list):
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)


def destroy_dirs(paths: list):
    for path in paths:
        shutil.rmtree(path)


def calc_dice(pred, target):
    if not pred.any():
        return 0
    if not target.any():
        return 0
    smooth = 1
    A = np.reshape(pred, (1, -1))  # Flatten
    B = np.reshape(target, (1, -1))
    intersection = (A * B).sum()
    return (2. * intersection + smooth) / (A.sum() + B.sum() + smooth)


def Dice(mask1_path, mask2_path):
    mask1 = cv2.imread(mask1_path, 0)
    _, mask1 = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY)
    mask1 = mask1 / 255
    mask2 = cv2.imread(mask2_path, 0)
    _, mask2 = cv2.threshold(mask2, 127, 255, cv2.THRESH_BINARY)
    mask2 = mask2 / 255

    dice = calc_dice(mask1, mask2)
    return dice


def create_video(frames_path, left_idx, right_idx, video_path):
    if not os.path.exists(video_path):
        os.mkdir(video_path)
    frames_list = sorted(os.listdir(frames_path))
    video_frames = frames_list[left_idx: right_idx + 1]
    for frame in video_frames:
        shutil.copy(os.path.join(frames_path, frame), os.path.join(video_path, frame))


def create_videos(frames_path, videos_path, radius):
    frames_list = sorted(os.listdir(frames_path))
    for idx in range(len(frames_list)):
        video_dir = os.path.join(videos_path, "{:04d}".format(idx))
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)

        left_index = max(0, idx - radius)
        right_index = min(len(frames_list) - 1, idx + radius)
        video_frames = frames_list[left_index: right_index + 1]
        for frame in video_frames:
            shutil.copy(os.path.join(frames_path, frame), os.path.join(video_dir, frame))


def extract_bboxes(mask_path):
    boxes = []
    centers = []
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        box = [x, y, x + w, y + h]
        boxes.append(box)
        center = [x + w / 2, y + h / 2]
        centers.append(center)
    return (num_labels, boxes, centers)


def calc_correlation(SQA_score_path, seg_dice_path):
    with open(SQA_score_path, 'r') as file1:
        score_dict = json.load(file1)
    with open(seg_dice_path, 'r') as file2:
        seg_dict = json.load(file2)

    seg_dice = list(seg_dict.values())
    scores = list(score_dict.values())

    pearson_res, _ = stats.pearsonr(seg_dice, scores)
    spearman_res, _ = stats.spearmanr(seg_dice, scores)
    kendall_res, _ = stats.kendalltau(seg_dice, scores)

    return (pearson_res, spearman_res, kendall_res)
