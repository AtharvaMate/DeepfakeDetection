import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications.xception import preprocess_input
YUNET_MODEL = "face_detection_yunet_2023mar.onnx"

face_detector = cv2.FaceDetectorYN.create(
    YUNET_MODEL,
    "",
    (320, 320),
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=1
)



IMG_SIZE = 224
NUM_FRAMES = 16
MODEL_PATH = hf_hub_download(
        repo_id="AtharvaMate/dfd_model",
        filename="dfd_model.keras"
    )

FAKE_THRESHOLD = 0.6
CONF_THRESHOLD = 70

st.set_page_config(
    page_title="Deepfake Video Detector",
    page_icon="🎥",
    layout="centered"
)

st.title("🎥 Deepfake Video Detection")
st.divider()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

def preprocess_frame(frame):
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = preprocess_input(frame)
    frame = np.expand_dims(frame, axis=0)
    return frame

def crop_face(frame):
    h, w, _ = frame.shape

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(bgr)

    if faces is None or len(faces) == 0:
        return None

    face = faces[0]
    x, y, fw, fh = map(int, face[:4])

    margin = 0.25
    x1 = max(0, int(x - margin * fw))
    y1 = max(0, int(y - margin * fh))
    x2 = min(w, int(x + fw * (1 + margin)))
    y2 = min(h, int(y + fh * (1 + margin)))

    cropped = frame[y1:y2, x1:x2]

    if cropped.size == 0:
        return None

    return cropped



def extract_frames(video_path, num_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return frames

    idxs = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()
    return frames

video_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov"]
)

if video_file:
    st.video(video_file)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    if st.button("🔍 Analyze Video", use_container_width=True):
        with st.spinner("Extracting frames and analyzing faces..."):

            frames = extract_frames(video_path)

            if len(frames) == 0:
                st.error("❌ Could not extract frames.")
                st.stop()

            probs = []

            for frame in frames:
                face = crop_face(frame)
                if face is None:
                    continue

                gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
                if gray.mean() < 40:
                    continue

                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                if blur_score < 50:
                    continue

                img = preprocess_frame(face)
                prob = float(model.predict(img, verbose=0)[0][0])
                probs.append(prob)

            if len(probs) < 5:
                st.warning("⚠️ Not enough face frames detected.")
                st.stop()

            probs = np.array(probs)

            top_k = np.sort(probs)[-max(3, len(probs)//4):]
            video_score = np.mean(top_k)

            fake_ratio = np.mean(probs > FAKE_THRESHOLD)

            if video_score >= FAKE_THRESHOLD and fake_ratio > 0.4:
                confidence = video_score * 100
                if confidence > CONF_THRESHOLD:
                    st.error("⚠️ Deepfake Video Detected")
                else:
                    st.warning("⚠️ Possible Deepfake Video Detected")
            else:
                confidence = (1 - video_score) * 100
                if confidence > CONF_THRESHOLD:
                    st.success("✅ Real Video")
                else:
                    st.info("ℹ️ Possibly Real Video")

            st.markdown(f"### Confidence: **{confidence:.2f}%**")

            if confidence < 75:
                st.warning("⚠️ Low confidence prediction")

        st.divider()
