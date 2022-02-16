import cv2
import torch
import numpy as np
from models.model_1 import Encoder, Decoder
import mediapipe as mp
import utils
import json


cap = cv2.VideoCapture(0)

encoder = Encoder()
decoder = Decoder()
encoder.load_state_dict(torch.load('../weights/encoder-1.pth'))
decoder.load_state_dict(torch.load('../weights/decoder-1.pth'))
encoder.to('cuda')
decoder.to('cuda')
encoder.eval()
decoder.eval()


with open('face_feature.json', 'r') as f:
    feature = json.load(f)

cnt = 0
with mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False,
) as face_mesh:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        results = utils.mp_landmarks([frame], face_mesh)
        key = cv2.waitKey(16)
        if results[0] is not None:
            for landmark in results[0]:
                cv2.circle(frame, (int(landmark[0] * 640), int(landmark[1] * 480)), 1, (255, 0, 0), -1)
            landmarks = utils.landmark_transpose(results)
            landmarks = torch.tensor(landmarks, dtype=torch.float32).to('cuda')
            features = encoder(landmarks)
            features = features[0].cpu().detach().numpy()
            for name, feats in feature.items():
                for feat in feats:
                    feat = np.array(feat)
                    if np.mean((feat - features) ** 2) < 2e-7:
                        print(cnt, name)
                        cnt += 1
        cv2.imshow('frame', frame)
        if key in [ord('q'), 27]:
            break
