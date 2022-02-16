import typing
import numpy as np
import cv2
import json
import mediapipe as mp
from types import SimpleNamespace

face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')


def face_box(imgs):
    for img in imgs:
        faces = face_cascade.detectMultiScale(
            image=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.2,
            minNeighbors=3,
        )
        try:
            x, y, w, h = faces[0]
            yield x, y, x + w, y + h
        except:
            yield None


def mp_landmarks(imgs: typing.Union[list, np.ndarray], solver: mp.solutions.face_mesh.FaceMesh = None) -> np.ndarray:
    imgs = np.array([cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs])
    results = []
    if solver is None:
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
        )
    else:
        face_mesh = solver
    for im in imgs:
        result = face_mesh.process(im)
        if not result.multi_face_landmarks:
            results.append(None)
            continue
        results.append([])
        for landmark in result.multi_face_landmarks[0].landmark:
            results[-1].append([landmark.x, landmark.y])
    if solver is None:
        face_mesh.close()
    return np.array(results)


def landmark_transpose(landmarks: np.ndarray, center_landmark_idx: int = 1) -> np.ndarray:
    landmarks = landmarks.copy()
    for i in landmarks:
        if not isinstance(i, np.ndarray):
            i = np.array(i.copy())
        i[:, 0] = i[:, 0] - i[center_landmark_idx, 0]
        i[:, 1] = i[:, 1] - i[center_landmark_idx, 1]
        i[:, :] *= np.array([1]) / np.sqrt(
            np.sum(np.square(i[0, :] - i[1, :])))
    return np.array([i.flatten() for i in landmarks])


class TrainConfig(SimpleNamespace):
    def __init__(self, **kwargs):
        self.device: str
        self.batch_size: int
        self.start_epoch: int
        self.end_epoch: int
        self.dataset_path: str
        self.encoder_path: str
        self.decoder_path: str
        super(TrainConfig, self).__init__(**kwargs)


def load_train_config(filename) -> TrainConfig:
    with open(filename, 'r') as f:
        data = json.load(f, object_hook=lambda x: TrainConfig(**x))
    return data


def write_train_config(filename, config: TrainConfig):
    with open(filename, 'w') as f:
        json.dump(config.__dict__, f, indent=2)
