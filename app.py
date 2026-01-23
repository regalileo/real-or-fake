import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
import io
import os # Import os module for path operations

# --- Konfigurasi Aplikasi Streamlit ---
st.set_page_config(
    page_title="AI vs Real Image Detector",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ AI vs Real Image Detector")
st.write("Unggah gambar untuk mendeteksi apakah itu gambar asli atau hasil dari AI.")

# --- DEBUGGING: Print current working directory and list files ---
st.write(f"Current working directory: {os.getcwd()}")
st.write(f"Files in current directory: {os.listdir('.')}")

# --- Preprocessing Gambar (harus sama dengan saat pelatihan) ---
# Normalisasi mean dan std dari ImageNet
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# --- Muat Model ONNX ---
@st.cache_resource
def load_onnx_model(model_path):
    try:
        session = ort.InferenceSession(model_path)
        return session
    except Exception as e:
        st.error(f"Gagal memuat model ONNX: {e}")
        st.error(f"Pastikan file '{model_path}' ada dan dapat diakses. Detail error: {e}") # Added more detailed error message
        return None

model_path = "mobilenet_v2_ai_real_embedded.onnx"
ort_session = load_onnx_model(model_path)

if ort_session:
    # --- Fungsi Prediksi ---
    def predict_image(image: Image.Image, session: ort.InferenceSession):
        # Terapkan transformasi
        input_tensor = transform(image).unsqueeze(0).numpy() # Tambahkan batch dimension dan konversi ke numpy

        # Siapkan input untuk ONNX Runtime
        ort_inputs = {session.get_inputs()[0].name: input_tensor}

        # Jalankan inferensi
        ort_outputs = session.run(None, ort_inputs)

        # Post-processing output
        output = ort_outputs[0]
        probabilities = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)

        class_labels = ["Real Image", "AI Generated Image"]
        predicted_class_idx = np.argmax(probabilities).item()
        predicted_label = class_labels[predicted_class_idx]
        confidence = probabilities[0, predicted_class_idx] * 100

        return predicted_label, confidence

    # --- UI Unggah Gambar ---
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Tampilkan gambar yang diunggah
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Gambar yang Diunggah', use_column_width=True)
        st.write("")

        if st.button("Deteksi Gambar"):
            with st.spinner('Mendeteksi...'):
                label, confidence = predict_image(image, ort_session)
                st.success(f"**Prediksi:** {label} (Keyakinan: {confidence:.2f}%)➡️")
            
else:
    st.warning("Model tidak dapat dimuat. Pastikan file 'mobilenet_v2_ai_real_embedded.onnx' ada di direktori yang sama.")
