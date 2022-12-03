import cv2
import numpy as np
import os
import torch


def center_crop(x, image_size):
    crop_h, crop_w = image_size
    h, w, _ = x.shape
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return x[max(0,j):min(h,j+crop_h), max(0,i):min(w,i+crop_w), :]


def get_input(scene_dir, idx):
    scene_path1 = os.path.join(scene_dir, 'Input/{:03d}-1'.format(idx + 1))
    scene_path2 = os.path.join(scene_dir, 'Input/{:03d}-2'.format(idx + 1))

    in1_L = cv2.imread(scene_path2 + '/1.jpg')
    in1_H = cv2.imread(scene_path1 + '/7.jpg')
    lab1 = cv2.imread(os.path.join(scene_dir, 'Label/{:03d}-1.jpg'.format(idx + 1)))

    in2_L = cv2.imread(scene_path1 + '/1.jpg')
    in2_H = cv2.imread(scene_path2 + '/7.jpg')
    lab2 = cv2.imread(os.path.join(scene_dir, 'Label/{:03d}-2.jpg'.format(idx + 1)))
    height, weith, _ = in1_L.shape
    j = int(height / 16.)
    i = int(weith / 16.)
    image_size = [j * 16, i * 16]
    in1_L = center_crop(in1_L, image_size)
    in1_H = center_crop(in1_H, image_size)
    in2_L = center_crop(in2_L, image_size)
    in2_H = center_crop(in2_H, image_size)
    lab1 = center_crop(lab1, image_size)
    lab2 = center_crop(lab2, image_size)
    return in1_L, in1_H, lab1, in2_L, in2_H, lab2


MAX_FEATURES = 5000
GOOD_MATCH_PERCENT = 0.15


def alignImages(im1, im2, img):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    #
    # # Draw top matches
    # imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    img_warped = cv2.warpPerspective(img, h, (width, height))

    return img_warped, h
