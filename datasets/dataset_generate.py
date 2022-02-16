import os
import cv2
import numpy as np
import mediapipe as mp

import utils


def generate_dataset(root = "raw/img_align_celeba/", start = 1, end = 202600, batch_size = 64):
    with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
         ) as face_mesh:
        for i in range((end - start) // batch_size):
            with open('landmarks.txt', "a") as f:
                imgs = []
                for j in range(i * batch_size, (i + 1) * batch_size):
                    imgs.append(cv2.imread(root + "%06d" % (j + start) + ".jpg"))
                landmarks = utils.mp_landmarks(imgs, face_mesh)
                landmarks = np.array([landmark for landmark in landmarks if landmark is not None])
                landmarks = utils.landmark_transpose(landmarks)
                for landmark in landmarks:
                    if landmark is not None:
                        f.write(' '.join(landmark.astype(str)) + "\n")
                print(f"batch {i + 1}/{(end - start) // batch_size}")
        with open('landmarks.txt', "a") as f:
            imgs = []
            for j in range((end - start) // batch_size * batch_size, end):
                imgs.append(cv2.imread(root + "%06d" % j + ".jpg"))
            landmarks = utils.mp_landmarks(imgs, face_mesh)
            landmarks = utils.landmark_transpose(landmarks)
            for landmark in landmarks:
                if landmark is not None:
                    f.write(' '.join(landmark.astype(str)) + "\n")


generate_dataset()
