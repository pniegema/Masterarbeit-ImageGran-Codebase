import os
import time

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.io import wavfile
import cv2
import dlib
import random
import matplotlib.pyplot as plt
import librosa



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")



def align_face(image, image_size=(112, 112)):
    """
    Debugging function to test face detection and alignment.
    Input_Size: (128, 96) ( height, width)
    """
    # Convert image to 8-bit RGB

    image_8bit = (image * 255).astype(np.uint8)
    gray = cv2.cvtColor(image_8bit, cv2.COLOR_RGB2GRAY)



    # Detect faces
    faces = detector(gray, 0)

    if len(faces) == 0:
        return None  # No face detected

    #Copy the image to draw on for debugging
    #image_8bit_copy = image_8bit.copy()

    # Draw bounding boxes around detected faces
    #for face in faces:
    #    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    #    cv2.rectangle(image_8bit_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the image with bounding boxes
    #plt.imshow(image_8bit_copy)
    #plt.title("Detected Faces")
    #plt.axis("off")
    #plt.show()

    # Get facial landmarks
    shape = predictor(gray, faces[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])


    # Draw landmarks on the image for debugging
    # for (x, y) in landmarks:
    #     cv2.circle(image_8bit_copy_2, (x, y), 2, (0, 0, 255), -1)
    #
    # # Display the image with landmarks
    # plt.imshow(image_8bit_copy_2)
    # #plt.title("Detected Landmarks")
    # plt.axis("off")
    # plt.show()

    # Corresponding canonical landmarks
    canonical_landmarks = np.array([
        [30.2946, 51.6963],  # Left eye
        [65.5318, 51.5014],  # Right eye
        [48.0252, 71.7366],  # Nose tip
        [33.5493, 92.3655],  # Left mouth corner
        [62.7299, 92.2041]  # Right mouth corner
    ], dtype=np.float32)

    canonical_landmarks[:, 0] *= (image_size[1] / 96)
    canonical_landmarks[:, 1] *= (image_size[0] / 112)






    selected_landmarks = np.array([
        landmarks[36],  # Left eye
        landmarks[45],  # Right eye
        landmarks[30],  # Nose tip
        landmarks[48],  # Left mouth corner
        landmarks[54]   # Right mouth corner
    ], dtype=np.float32)

    #Plot the selected landmarks for debugging
    # for (x, y) in selected_landmarks:
    #     print(x, y)
    #     cv2.circle(image_8bit_copy_2, (int(x), int(y)), 2, (255, 0, 0), -1)
    # plt.imshow(image_8bit_copy_2)
    # #plt.title("Selected Landmarks")
    # plt.axis("off")
    # plt.show()
    #
    # for (x, y) in canonical_landmarks:
    #     print(x, y)
    #     cv2.circle(image_8bit_copy_2, (int(x), int(y)), 2, (255, 0, 0), -1)
    # plt.imshow(image_8bit_copy_2)
    # #plt.title("Canonical Landmarks")
    # plt.axis("off")
    # plt.show()

    transformation_matrix = procrustes_align(selected_landmarks, canonical_landmarks)

    if transformation_matrix is None:
        print("Alignment failed")
        return None

    aligned_face = cv2.warpAffine(image_8bit, transformation_matrix, (image_size[1], image_size[0]),
                                  flags=cv2.INTER_LINEAR)

    # Convert back to [0, 1] range
    aligned_face = aligned_face.astype(np.float32) / 255.0

    #Plot the aligned face for debugging
    #plt.imshow(aligned_face)
    #plt.title("Aligned Face")
    #plt.axis("off")
    #plt.show()

    return aligned_face


def procrustes_align(src, dst):
    """
    Align src points to dst using Procrustes analysis.
    This method reduces rotation distortion.
    """
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)

    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    u, s, vt = np.linalg.svd(dst_centered.T @ src_centered)
    rotation = u @ vt

    scale = np.sum(s) / np.sum(src_centered ** 2)

    transform_matrix = np.hstack([scale * rotation, (dst_mean - scale * (rotation @ src_mean)).reshape(2, 1)])

    return transform_matrix

class GRIDDataset(Dataset):
    def __init__(self, root_dir, audio_length=10000, image_size=(112, 112)):

        self.root_dir = root_dir
        self.audio_length = audio_length
        self.image_size = image_size
        self.speakers = os.listdir(root_dir)
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for speaker in self.speakers:
            speaker_dir = os.path.join(self.root_dir, speaker, "video")
            for video_file in os.listdir(speaker_dir):
                if video_file.endswith(".mpg"):
                    video_path = os.path.join(speaker_dir, video_file)
                    audio_path = os.path.join(self.root_dir, speaker, "audio", video_file.replace(".mpg", ".wav"))

                    data.append((video_path, audio_path))
        return data

    def _load_audio(self, audio_path):
        # Load audio waveform
        sample_rate, waveform = wavfile.read(audio_path)
        #Normalize the audio
        waveform = waveform / 32768.0
        return waveform, sample_rate

    def _load_video_frames(self, video_path, frame_indices):
        cap = cv2.VideoCapture(video_path)
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Jump to frame
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0  # Normalize
            frames.append(frame)
        cap.release()
        if len(frames) < len(frame_indices):
            print(f"Warning: Could not load all frames for {video_path}")
            return None
        return np.array(frames)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, retry_count=0, max_retries=10):
        """
        Args:
            idx: Index of the sample to retrieve.
            retry_count: Number of retries attempted so far.
            max_retries: Maximum number of retries before skipping the sample.
        Returns:
            audios: Audio windows as a tensor of shape (batch, 1 , audio_length).
            current_frame: Current frame as a tensor of shape (3, 112, 112).
            next_frames: Next frames as a tensor of shape (3, 112, 112).
        """
        overall_start = time.time()

        if retry_count >= max_retries:
            raise RuntimeError(f"Failed to find a valid sample after {max_retries} retries.")

        video_path, audio_path = self.data[idx]


        audio_waveform, sample_rate = self._load_audio(audio_path)


        # Randomly select identity_frame and a different training frame from the video
        frame_idx = np.random.randint(12, 50)
        sample_idx = frame_idx + random.randint(1, 5) * 3

        video_frames = self._load_video_frames(video_path, [frame_idx, sample_idx])

        #Plot an example of the video frames for debugging
        #plt.imshow(video_frames[0])
        #plt.title("Identity Frame")
        #plt.axis("off")
        #plt.show()


        if video_frames is None:
            return self.__getitem__(np.random.randint(0, len(self.data)), retry_count + 1, max_retries)


        aligned_identity_frame = align_face(video_frames[0], self.image_size)
        if aligned_identity_frame is None:
            # If no face is detected, retry with a new random sample
            return self.__getitem__(np.random.randint(0, len(self.data)), retry_count + 1, max_retries)


        training_frame = video_frames[1]
        aligned_training_frame = align_face(training_frame, self.image_size)


        start = (sample_idx) * int(sample_rate / 25) - self.audio_length // 2
        end = start + self.audio_length
        sampled_audio = audio_waveform[start:end]
        if len(sampled_audio) < self.audio_length:
            padding = self.audio_length - len(sampled_audio)
            sampled_audio = np.pad(sampled_audio, (0, padding), mode="constant")



        n_mfcc = 12
        n_fft = 1250
        hop_length = 500
        MFCC = extract_mfcc(sampled_audio, sample_rate, n_mfcc, n_fft, hop_length)
        MFCC = (MFCC - np.mean(MFCC)) / np.std(MFCC)


        # Measure conversion to PyTorch tensors
        audio_mfcc = torch.tensor(MFCC, dtype=torch.float32).unsqueeze(0)
        if aligned_identity_frame is None or aligned_training_frame is None:
            return self.__getitem__(np.random.randint(0, len(self.data)), retry_count + 1, max_retries)

        identity_frame = torch.tensor(aligned_identity_frame, dtype=torch.float32).permute(2, 0, 1)
        training_frame = torch.tensor(np.array(aligned_training_frame), dtype=torch.float32).permute(2, 0, 1)



        return audio_mfcc, identity_frame, training_frame

def get_landmarks(image):
    image_8bit = (image * 255).astype(np.uint8)
    gray = cv2.cvtColor(image_8bit, cv2.COLOR_RGB2GRAY)

    faces = detector(gray, 1)

    if len(faces) == 0:
        #print("No face detected")
        return None  # No face detected

    # Get facial landmarks
    shape = predictor(gray, faces[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])

    # # Draw landmarks on the image for debugging
    # for i, (x, y) in enumerate(landmarks):
    #     if i in [49, 50, 52, 53, 59, 58, 56, 55, 48, 54]:
    #         cv2.circle(image_8bit, (x, y), 1, (255, 0, 0), -1)
    #     else:
    #         cv2.circle(image_8bit, (x, y), 1, (0, 0, 255), -1)
    #
    # # Display the image with landmarks
    # plt.imshow(image_8bit)
    # plt.title("Detected Landmarks")
    # plt.show()

    #Extract the mouth points
    left = landmarks[48]
    right = landmarks[54]
    top_left = landmarks[49]
    top_left_middle = landmarks[50]
    top_right_middle = landmarks[52]
    top_right = landmarks[53]
    bottom_left = landmarks[59]
    bottom_left_middle = landmarks[58]
    bottom_right_middle = landmarks[56]
    bottom_right = landmarks[55]

    #Return points in the order of the mouth
    mouth_points = [left, top_left, top_left_middle, top_right_middle, top_right, right, bottom_right, bottom_right_middle, bottom_left_middle, bottom_left]
    return mouth_points

def extract_mfcc(audio, sr, n_mfcc=12, n_fft=1250, hop_length=500):
    """
    Extracts MFCC features from raw audio.

    Args:
        audio (numpy.ndarray): Raw audio waveform.
        sr (int): Sample rate of the audio.
        n_mfcc (int): Number of MFCC coefficients to return.
        n_fft (int): FFT window size (default 1250 for 25ms window at 50kHz).
        hop_length (int): Hop size (default 500 for 10ms hop at 50kHz).

    Returns:
        numpy.ndarray: MFCC features of shape (n_mfcc, time_steps).
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfcc


def plot_mfcc(mfcc, sr, hop_length):
    """
    Plots the MFCC features.

    Args:
        mfcc (numpy.ndarray): MFCC feature matrix (n_mfcc, time_steps).
        sr (int): Sample rate of the audio.
        hop_length (int): Hop size for MFCC computation.
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, sr=sr, hop_length=hop_length, x_axis="time", cmap="viridis")
    plt.colorbar(label="MFCC Coefficients")
    plt.title("MFCC")
    plt.xlabel("Time (s)")
    plt.ylabel("MFCC Coefficients")
    plt.show()


def plot_landmarks_image(path):
    """
    plots the facial landmarks in an image.

    Args:
        path (str): Path to the image file.

    Returns:
        numpy.ndarray: Detected facial landmarks.
    """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    faces = detector(gray, 1)

    if len(faces) == 0:
        print("No face detected")
        return None  # No face detected

    # Get facial landmarks
    shape = predictor(gray, faces[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])

    # Draw all landmarks on the image
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 1, (235, 250, 250), -1)

    # Save the image
    plt.imsave("landmarks.png", image)




def main():
    #Check the dataset funconality

    root_dir = r"C:\Users\pnieg\Documents\Masterarbeit\GRIDRAW"

    start = time.time()
    dataset = GRIDDataset(root_dir)
    print("Time to load dataset:", time.time() - start)


    start = time.time()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    print("Time to load dataloader:", time.time() - start)



    #Get and print the first 8 Identity Frames
    for audio, identity_frame, training_frame in dataloader:
        for i in range(8):
            #plot the mfcc
            plot_mfcc(audio[i].squeeze().cpu().detach().numpy(), 50000, 500)



        break


    #Uncomment for detailed plotting of dataset samples
    # for audio, identity_frame, training_frame in dataloader:
    #     #Plot the first batch
    #     for i in range(8):
    #         plt.figure(figsize=(10, 5))
    #         plt.subplot(1, 3, 1)
    #         plt.imshow(identity_frame[i].permute(1, 2, 0).cpu().detach().numpy())
    #         plt.title("Identity Frame")
    #         plt.axis("off")
    #         plt.subplot(1, 3, 2)
    #         plt.imshow(training_frame[i].permute(1, 2, 0).cpu().detach().numpy())
    #         plt.title("Training Frame")
    #         plt.axis("off")
    #         plt.show()
    #         plot_mfcc(audio[i].squeeze().cpu().detach().numpy(), 50000, 500)
    #     print("Audio shape:", audio.shape)
    #     print("Identity Frame shape:", identity_frame.shape)
    #     print("Training Frame shape:", training_frame.shape)
    #     break




if __name__ == "__main__":
    main()


