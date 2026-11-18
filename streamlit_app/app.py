import streamlit as st
import numpy as np
import json
from pathlib import Path
from PIL import Image
import tensorflow as tf

# ================= KONFIG =================
st.set_page_config(page_title="Klasifikasi Kondisi Jalan",
                   page_icon="🛣️", layout="centered")

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model_terbaik.keras"
INFO_PATH  = BASE / "model_info.json"

@st.cache_resource
def load_model_and_info():
    model = tf.keras.models.load_model(MODEL_PATH)
    info  = json.loads(INFO_PATH.read_text())
    return model, info

model, info = load_model_and_info()
CLASS_NAMES = info["class_names"]          # ["berlubang","normal","rusak"]
IMG_SIZE    = info["img_size"]             # 224
USE_RESCALE = info["use_rescale"]          # False untuk EfficientNetB0

# ================= UI =================
st.title("🛣️ Klasifikasi Kondisi Jalan")
st.markdown(
    "Unggah gambar permukaan jalan, sistem akan mengklasifikasikan "
    "ke dalam **Berlubang**, **Normal**, atau **Rusak**."
)
st.caption(f"Model: {info['model_name']} · Akurasi uji: "
           f"{info['metrics']['accuracy']*100:.2f}%")

uploaded = st.file_uploader("Pilih gambar (JPG/PNG)",
                            type=["jpg","jpeg","png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns([1,1])
    with col1:
        st.image(img, caption="Gambar diunggah", use_container_width=True)

    # ---------- Preprocessing ----------
    arr = np.array(img.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    if USE_RESCALE:
        arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)      # (1,224,224,3)

    # ---------- Prediksi ----------
    with st.spinner("Menganalisis..."):
        prob = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(prob))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(prob[pred_idx]) * 100

    # ---------- Hasil ----------
    with col2:
        st.subheader("Hasil Prediksi")
        if pred_label == "normal":
            st.success(f"✅ **{pred_label.upper()}**")
        elif pred_label == "berlubang":
            st.error(f"🕳️ **{pred_label.upper()}**")
        else:
            st.warning(f"⚠️ **{pred_label.upper()}**")
        st.metric("Confidence", f"{confidence:.2f}%")

        st.markdown("**Probabilitas per kelas:**")
        for name, p in sorted(zip(CLASS_NAMES, prob), key=lambda x:-x[1]):
            st.progress(float(p), text=f"{name}: {p*100:.2f}%")

    st.divider()
    st.caption("⚠️ Hasil bersifat estimasi. Untuk keputusan teknis, "
               "tetap diperlukan verifikasi lapangan.")
