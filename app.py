# FastAPI visual search for WildTrack (SeaTurtleID)
import os, io, csv, base64, json
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn
import tensorflow as tf
from tensorflow import keras

OUTPUT_DIR = r"C:\Users\sagni\Downloads\WildTrack"
DATA_DIR   = r"C:\Users\sagni\Downloads\WildTrack\archive\turtles-data\data"
EMB_NPY    = r"C:\Users\sagni\Downloads\WildTrack\embeddings.npy"
META_CSV   = r"C:\Users\sagni\Downloads\WildTrack\meta.csv"
FAISS_IDX  = r"C:\Users\sagni\Downloads\WildTrack\index.faiss"
NP_IDX     = r"C:\Users\sagni\Downloads\WildTrack\index.npz"
MODEL_KERAS = r"C:\Users\sagni\Downloads\WildTrack\model.keras"
MODEL_H5    = r"C:\Users\sagni\Downloads\WildTrack\model.h5"
IMG_SIZE    = [224, 224]

# ---- L2Normalize layer (for loading) ----
try:
    from tensorflow.keras.utils import register_keras_serializable
except Exception:
    try:
        from keras.utils import register_keras_serializable
    except Exception:
        def register_keras_serializable(package="WildTrack"):
            def deco(obj): return obj
            return deco

@register_keras_serializable(package="WildTrack")
class L2Normalize(keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
    def call(self, x):
        return tf.math.l2_normalize(x, axis=self.axis)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"axis": self.axis})
        return cfg

# ---- Load meta and embeddings ----
paths, labels = [], []
with open(META_CSV, newline="", encoding="utf-8") as fcsv:
    r = csv.DictReader(fcsv)
    for row in r:
        paths.append(row["path"])
        labels.append(row["label"])

if os.path.exists(EMB_NPY):
    embeddings = np.load(EMB_NPY).astype(np.float32)
else:
    raise FileNotFoundError("Missing embeddings.npy. Run the index builder first.")

use_faiss = False
try:
    import faiss
    if os.path.exists(FAISS_IDX):
        index = faiss.read_index(FAISS_IDX)
        use_faiss = True
except Exception:
    if os.path.exists(NP_IDX):
        npz = np.load(NP_IDX)
        # embeddings already loaded

# ---- Load classifier & build embedding model ----
def load_classifier():
    custom = {"L2Normalize": L2Normalize}
    if os.path.exists(MODEL_KERAS):
        try:
            return keras.models.load_model(MODEL_KERAS, custom_objects=custom)
        except Exception as e:
            print("[WARN] model.keras load failed:", e)
    if os.path.exists(MODEL_H5):
        try:
            return keras.models.load_model(MODEL_H5, compile=False, custom_objects=custom)
        except Exception as e:
            print("[WARN] model.h5 load failed:", e)
    raise FileNotFoundError("No model file found.")

clf = load_classifier()

def build_embedding_model(classifier):
    # Prefer l2norm if present
    try:
        l = classifier.get_layer("l2norm")
        return keras.Model(classifier.inputs, l.output)
    except Exception:
        pass
    # Pre-softmax fallback
    softmax_layer = None
    for l in classifier.layers[::-1]:
        if isinstance(l, keras.layers.Dense) and getattr(l, "activation", None) == keras.activations.softmax:
            softmax_layer = l
            break
        if l.name.lower() == "softmax":
            softmax_layer = l
            break
    if softmax_layer is None:
        return keras.Model(classifier.inputs, classifier.layers[-1].output)
    emb_tensor = softmax_layer.input
    inp = classifier.inputs
    x = keras.Model(inp, emb_tensor)(inp)
    x = tf.math.l2_normalize(x, axis=-1)
    return keras.Model(inp, x)

emb_model = build_embedding_model(clf)

# ---- Helpers ----
def load_and_preprocess_bytes(b: bytes):
    img = Image.open(io.BytesIO(b)).convert("RGB")
    img = img.resize(tuple(IMG_SIZE), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    return arr

def encode_thumb(path, max_side=256):
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = max_side / max(w, h)
        if scale < 1:
            img = img.resize((int(w*scale), int(h*scale)), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None

# ---- Search core ----
def search_vector(qemb: np.ndarray, top_k=5):
    if use_faiss:
        import faiss
        D, I = index.search(qemb[None, :].astype(np.float32), top_k)
        sims = D[0].tolist()
        inds = I[0].tolist()
    else:
        sims_all = embeddings @ qemb.astype(np.float32)
        inds = np.argsort(sims_all)[::-1][:top_k].tolist()
        sims = [float(sims_all[i]) for i in inds]
    out = []
    for rnk,(i,s) in enumerate(zip(inds, sims), 1):
        out.append({
            "rank": rnk,
            "index": int(i),
            "path": paths[i],
            "label": labels[i],
            "similarity": float(s),
            "thumbnail": encode_thumb(paths[i])
        })
    return out

# ---- FastAPI app ----
app = FastAPI(title="WildTrack Search API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    html = (Path(OUTPUT_DIR) / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)

@app.post("/search")
async def do_search(file: UploadFile = File(...), top_k: int = Form(5)):
    b = await file.read()
    arr = load_and_preprocess_bytes(b)
    qemb = emb_model.predict(arr[None, ...], verbose=0)[0]
    qemb = qemb / (np.linalg.norm(qemb) + 1e-12)
    results = search_vector(qemb, top_k=top_k)
    return JSONResponse({"results": results})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
