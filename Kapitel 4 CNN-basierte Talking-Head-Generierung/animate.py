import torch
import torch._dynamo
import numpy as np
import cv2
import librosa
from CNN_basiertes_Model import Speech2Vid
import dlib
import moviepy.editor as mpy

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
    faces = detector(gray, 1)

    if len(faces) == 0:
        print("No face detected")
        return None  # No face detected

    # # Draw bounding boxes around detected faces for debugging and viszualization
    # for face in faces:
    #     x, y, w, h = face.left(), face.top(), face.width(), face.height()
    #     cv2.rectangle(image_8bit, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    # # Display the image with bounding boxes
    # plt.imshow(image_8bit)
    # plt.title("Detected Faces")
    # plt.show()

    # Get facial landmarks
    shape = predictor(gray, faces[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])

    # # Draw landmarks on the image for debugging and viszualization
    # for (x, y) in landmarks:
    #     cv2.circle(image_8bit, (x, y), 2, (0, 0, 255), -1)
    #
    # # Display the image with landmarks
    # plt.imshow(image_8bit)
    # plt.title("Detected Landmarks")
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

    transformation_matrix = procrustes_align(selected_landmarks, canonical_landmarks)

    if transformation_matrix is None:
        print("Alignment failed")
        return None

    aligned_face = cv2.warpAffine(image_8bit, transformation_matrix, (image_size[1], image_size[0]),
                                  flags=cv2.INTER_LINEAR)

    # Convert back to [0, 1] range
    aligned_face = aligned_face.astype(np.float32) / 255.0

    return aligned_face


def procrustes_align(src, dst):
    """
    Align src points to dst using Procrustes analysis.
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

def animate_image_mfcc_stepwise(model, audio_path, identity_frame, output_video="output.mp4", window_size=10000, step_size=2000, fps = 25):
    """

    :param model: Model that takes a MFCC of an audio and an identity frame and generates a corresponding to the identity frmae speaking the audio
    :param audio: Path to the Raw Audio Waveform. Needs to be Split into windows of window_size and transformed into MFCCs, padd with zeros if necessary
    :param identity_frame: Path to the identity frame
    :param window_size: Size of the audio windows
    :param step_size: Size between windows
    :return: All animated frames / Animation
    """
    audio, sr = librosa.load(audio_path, sr=50000)
    frame = cv2.imread(identity_frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0  # Convert to RGB and normalize
    identity_frame = align_face(frame, image_size=(112, 112))
    identity_frame = torch.tensor(identity_frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
    print("Identity Frame Shape:", identity_frame.shape)
    print("Audio Shape:", audio.shape)


    frames = []

    for i in range(0, len(audio), step_size):
        if i - int(window_size / 2) < 0:
            audio_window = audio[:int(window_size / 2)]
            # Pad left side with zeros
            audio_window = np.pad(audio_window, (window_size - len(audio_window), 0), mode="constant")
        elif i + int(window_size / 2) > len(audio):
            audio_window = audio[-int(window_size / 2):]
            # Pad right side with zeros
            audio_window = np.pad(audio_window, (0, window_size - len(audio_window)), mode="constant")
        else:
            audio_window = audio[i - int(window_size / 2):i + int(window_size / 2)]

        audio_window = extract_mfcc(audio_window, sr)
        audio_window = torch.tensor(audio_window).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            #Use last frame of frames and audio_window to generate next frame

            # If no frames have been generated yet, use the identity frame as input
            if len(frames) == 0:
                input_frame = identity_frame
            else:
                input_frame = torch.tensor(frames[-1]).permute(2, 0, 1).unsqueeze(0).float().to(device) # Use the last generated frame as the identity frame
            generated_frame = model(audio_window, input_frame)


            # generated_frame = deblurr(generated_frame)

        generated_frame = generated_frame.squeeze().permute(1, 2, 0).cpu().numpy()
        generated_frame = (generated_frame * 255).astype(np.uint8)  # Convert to uint8 for video
        frames.append(generated_frame)

    # Save as video
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_video = "temp_video.mp4"
    video_writer = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

    for frame in frames:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video_writer.release()

    # Add audio to the video
    video_clip = mpy.VideoFileClip(temp_video)
    audio_clip = mpy.AudioFileClip(audio_path)
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(output_video, codec="libx264", fps=fps, audio_codec="aac")

    print(f"Video saved as {output_video}")

    return frames



def animate_image_mfcc(model, audio_path, identity_frame, output_video="output.mp4", window_size=10000, step_size=2000, fps = 25):
    """

    :param model: Model that takes a MFCC of an audio and an identity frame and generates a corresponding to the identity frmae speaking the audio
    :param audio: Path to the Raw Audio Waveform. Needs to be Split into windows of window_size and transformed into MFCCs, padd with zeros if necessary
    :param identity_frame: Path to the identity frame
    :param window_size: Size of the audio windows
    :param step_size: Size between windows
    :return: All animated frames / Animation
    """
    audio, sr = librosa.load(audio_path, sr=50000)
    frame = cv2.imread(identity_frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0  # Convert to RGB and normalize
    identity_frame = align_face(frame, image_size=(112, 112))
    identity_frame = torch.tensor(identity_frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
    print("Identity Frame Shape:", identity_frame.shape)
    print("Audio Shape:", audio.shape)

    frames = []


    for i in range(0, len(audio), step_size):
        if i - int(window_size/2) < 0:
            audio_window = audio[:int(window_size/2)]
            #Pad left side with zeros
            audio_window = np.pad(audio_window, (window_size - len(audio_window), 0), mode="constant")
        elif i + int(window_size/2) > len(audio):
            audio_window = audio[-int(window_size/2):]
            #Pad right side with zeros
            audio_window = np.pad(audio_window, (0, window_size - len(audio_window)), mode="constant")
        else:
            audio_window = audio[i-int(window_size/2):i+int(window_size/2)]

        audio_window = extract_mfcc(audio_window, sr)
        audio_window = torch.tensor(audio_window).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():

            generated_frame = model(audio_window, identity_frame)

        generated_frame = generated_frame.squeeze().permute(1, 2, 0).cpu().numpy()
        generated_frame = (generated_frame * 255).astype(np.uint8)# Convert to uint8 for video
        frames.append(generated_frame)

    # Save as video
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_video = "temp_video.mp4"
    video_writer = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

    for frame in frames:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video_writer.release()

    # Add audio to the video
    video_clip = mpy.VideoFileClip(temp_video)
    audio_clip = mpy.AudioFileClip(audio_path)
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(output_video, codec="libx264", fps=fps, audio_codec="aac")

    print(f"Video saved as {output_video}")

    return frames



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






if __name__ == '__main__':

    model= Speech2Vid().to(device)
    model.load_state_dict(torch.load('YourModelPath.pth', weights_only=True))
    identity_frame_path = "YourIdentityFramePath.jpg"
    audio_path = "YourAudioPath.wav"
    output_video_path = "YourOutputVideoPath.mp4"

    frames = animate_image_mfcc(model=model, audio_path=audio_path, identity_frame=identity_frame_path, window_size=10000, step_size=2000, output_video=output_video_path, fps=25)

    print("Frames:", len(frames))


