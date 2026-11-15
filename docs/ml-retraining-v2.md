# PhytoCast — Model Retrain v2 (bigger dataset + our own Syngonium)

**What changes from v1**
- Dataset: PlantVillage (54,305 images) → **New Plant Diseases Dataset, ~87,900 images, 38 classes** (Kaggle: `vipoooool/new-plant-diseases-dataset`). It is an augmented version of PlantVillage, pre-split 80/20 into train/valid.
- **New 39th class: `Syngonium___healthy`** — photos of our own arrowhead plant.
- Training: two phases (train the new head, then fine-tune the top of MobileNetV2), plus on-the-fly augmentation and class weighting.
- **Real-world test: PlantDoc** — 2,598 field photos (Google/Ecosia images of leaves in real conditions), never used for training. This measures how the model does outside the lab.
- Testing: a confidence threshold, so low-confidence guesses say "not sure" instead of naming a wrong disease.

**Honest scope:** no public dataset of diseased Syngonium exists. The model learns what a *healthy* Syngonium looks like; it cannot diagnose Syngonium diseases.

**Time:** ~1 hour total on the free Colab GPU, most of it unattended.

---

## Before you start — photograph the plant (~15 min)

Take **200–300 photos** of the arrowhead plant with a phone. Variety matters more than count:
- Mostly **single leaves, close-up** (the dataset is single-leaf images) — both the top and underside.
- Different **leaves**, **angles**, and **distances**.
- Different **light**: daylight, room light, near a window.
- Different **backgrounds**: some on plain paper/table (like the dataset), some with the pot and room behind.

Put all photos in one folder, zip it as **`syngonium.zip`**. (Photos only — no subfolders needed.) Photos must be **JPG or PNG** — iPhones save HEIC by default, so switch the camera to "Most Compatible" or convert first.

---

## Colab setup
**File → New notebook**, then **Runtime → Change runtime type → T4 GPU → Save.** Run each cell in order.

### Cell 1 — Check the GPU
```python
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))
```
Expect a GPU device in the list. If it's empty, redo the runtime step.

### Cell 2 — Download the dataset from Kaggle
```python
KAGGLE_TOKEN = "paste your KGAT_... token here"

import os
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
with open(os.path.expanduser("~/.kaggle/access_token"), "w") as f:
    f.write(KAGGLE_TOKEN)
os.chmod(os.path.expanduser("~/.kaggle/access_token"), 0o600)

!pip install -q kaggle
!kaggle datasets download -d vipoooool/new-plant-diseases-dataset -p /content/npd --unzip
```
~2.7 GB, a few minutes. On a 403 error, open the dataset page on kaggle.com, accept any rules banner, re-run.

### Cell 3 — Find the train/valid folders
```python
!find /content/npd -maxdepth 4 -type d \( -name train -o -name valid \)
```
The download contains the dataset **twice** (a capitalised and a lowercase top folder) — use the capitalised pair. Then run:
```python
TRAIN_DIR = "/content/npd/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
VALID_DIR = "/content/npd/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"
!rm -rf "/content/npd/new plant diseases dataset(augmented)"   # optional: delete the duplicate copy
```

### Cell 4 — Add our Syngonium photos as a new class (OPTIONAL)

**Skip this cell entirely** to train on the 87,900-image dataset alone (38 classes). Cells 5–11 work either way — just skip the Syngonium line at the end of Cell 9.

**Faster than photos:** record a **60-second video** of the plant instead (slowly circle it, get close to individual leaves, vary the light). Upload the video and run this instead of the zip upload below — it pulls ~300 frames automatically:
```python
import cv2
uploaded = files.upload()                        # pick the video
video = list(uploaded)[0]
os.makedirs("/content/syngonium_raw", exist_ok=True)
cap = cv2.VideoCapture(video)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
step = max(1, total // 300)
i = saved = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    if i % step == 0:
        cv2.imwrite(f"/content/syngonium_raw/frame_{saved}.jpg", frame); saved += 1
    i += 1
print("frames saved:", saved)
```
Then run the rest of this cell from `photos = [...]` onward (skip its upload/unzip lines). Note: frames from one video are very similar to each other, so the Syngonium accuracy will be even more optimistic than with photos.

Photo-zip version:
```python
from google.colab import files
import zipfile, glob, random, shutil

uploaded = files.upload()          # pick syngonium.zip
with zipfile.ZipFile("syngonium.zip") as z:
    z.extractall("/content/syngonium_raw")

photos = [p for p in glob.glob("/content/syngonium_raw/**/*", recursive=True)
          if p.lower().endswith((".jpg", ".jpeg", ".png"))]
random.seed(123)
random.shuffle(photos)
split = int(len(photos) * 0.8)

for subset, items in (("train", photos[:split]), ("valid", photos[split:])):
    dest = os.path.join(TRAIN_DIR if subset == "train" else VALID_DIR, "Syngonium___healthy")
    os.makedirs(dest, exist_ok=True)
    for i, p in enumerate(items):
        shutil.copy(p, os.path.join(dest, f"syn_{subset}_{i}{os.path.splitext(p)[1].lower()}"))

print(f"Syngonium: {split} train, {len(photos) - split} valid")
```

### Cell 4b — Save progress to Google Drive (protects against disconnects)
```python
from google.colab import drive, files
drive.mount('/content/drive')
CKPT = '/content/drive/MyDrive/phytocast_v2_best.keras'
ckpt = tf.keras.callbacks.ModelCheckpoint(CKPT, monitor='val_accuracy', save_best_only=True)
```
If Colab disconnects later: re-run Cells 1–3 and 5, then
`model = tf.keras.models.load_model(CKPT)` (after re-mounting Drive and `import numpy as np`) and jump to Cell 9 — run it from `y_true, y_pred = [], []` since the training history is gone.

### Cell 5 — Load the images and balance the classes
```python
IMG_SIZE = 224
BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE

ds_train = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH, shuffle=True, seed=123)
ds_val = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH, shuffle=False)

class_names = ds_train.class_names
num_classes = len(class_names)
print("Classes:", num_classes)

# Our Syngonium class has a few hundred photos vs ~2,000 per crop class,
# so weight it up so the model doesn't just ignore it.
counts = [len(os.listdir(os.path.join(TRAIN_DIR, c))) for c in class_names]
total = sum(counts)
class_weight = {i: total / (num_classes * n) for i, n in enumerate(counts)}

def prep(image, label):
    return tf.keras.applications.mobilenet_v2.preprocess_input(image), label

ds_train = ds_train.map(prep, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
ds_val   = ds_val.map(prep, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
```
Expect **39 classes**.

### Cell 6 — Build the model (with augmentation)
```python
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    # No RandomContrast: in Keras 3 it clips to 0..255, which wrecks images
    # already scaled to -1..1 by preprocess_input (caused a 72% model).
])

base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = augment(inputs)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
Augmentation only runs during training — it randomly flips/rotates/zooms images so the model copes better with real, messy photos.

**Epochs:** on the free T4, one epoch over ~70k images takes ~4–8 min. If short on time, use `epochs=3` in Cell 7 and `epochs=2` in Cell 8.

### Cell 7 — Phase 1: train the new head (~20–30 min)
```python
h1 = model.fit(ds_train, validation_data=ds_val, epochs=5, class_weight=class_weight, callbacks=[ckpt])
```

### Cell 8 — Phase 2: fine-tune the top of MobileNetV2 (~20–30 min)
```python
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False     # only the last 30 layers adapt

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
h2 = model.fit(ds_train, validation_data=ds_val, epochs=5, class_weight=class_weight, callbacks=[ckpt])
```
The very low learning rate matters — it nudges the pretrained layers instead of wrecking them.

### Cell 9 — Results: overall and for our plant
```python
import numpy as np
import matplotlib.pyplot as plt

acc = h1.history['accuracy'] + h2.history['accuracy']
val = h1.history['val_accuracy'] + h2.history['val_accuracy']
plt.plot(acc, label='train'); plt.plot(val, label='validation')
plt.axvline(len(h1.history['accuracy']) - 0.5, ls='--', c='gray', label='fine-tune starts')
plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.legend(); plt.title('Model accuracy'); plt.show()

y_true, y_pred = [], []
for images, labels in ds_val:
    y_pred.extend(np.argmax(model.predict(images, verbose=0), axis=1))
    y_true.extend(labels.numpy())
y_true, y_pred = np.array(y_true), np.array(y_pred)

print(f"Overall validation accuracy: {100 * (y_true == y_pred).mean():.1f}%")
s = class_names.index("Syngonium___healthy")
mask = y_true == s
print(f"Syngonium___healthy: {100 * (y_pred[mask] == s).mean():.1f}% of {mask.sum()} held-out photos")
```
**Screenshot the graph and both numbers — they go in the report.**

Note: the Syngonium validation photos come from the same plant and photo session as the training photos, so that number will be optimistic. Say so if asked. (Skipped Cell 4? Delete the last three lines of this cell.)

### Cell 9b — Real-world test on PlantDoc
PlantDoc's class names differ from ours (e.g. `Tomato Early blight leaf` vs `Tomato___Early_blight`), so this cell maps the matching ones and skips the rest, printing what it did.
```python
!git clone -q https://github.com/pratikkayal/PlantDoc-Dataset /content/plantdoc

from tensorflow.keras.utils import load_img, img_to_array

# PlantDoc folder (normalised) -> (crop word, disease word) found in OUR class name
MAP = {
    "apple scab leaf": ("apple", "scab"),        "apple leaf": ("apple", "healthy"),
    "apple rust leaf": ("apple", "rust"),
    "bell pepper leaf": ("pepper", "healthy"),   "bell pepper leaf spot": ("pepper", "bacterial"),
    "blueberry leaf": ("blueberry", "healthy"),  "cherry leaf": ("cherry", "healthy"),
    "corn gray leaf spot": ("corn", "gray"),     "corn leaf blight": ("corn", "blight"),
    "corn rust leaf": ("corn", "rust"),          "peach leaf": ("peach", "healthy"),
    "potato leaf early blight": ("potato", "early"), "potato leaf late blight": ("potato", "late"),
    "potato leaf": ("potato", "healthy"),        "raspberry leaf": ("raspberry", "healthy"),
    "soyabean leaf": ("soybean", "healthy"),     "squash powdery mildew leaf": ("squash", "powdery"),
    "strawberry leaf": ("strawberry", "healthy"),
    "tomato leaf": ("tomato", "healthy"),        "tomato early blight leaf": ("tomato", "early"),
    "tomato leaf late blight": ("tomato", "late"), "tomato septoria leaf spot": ("tomato", "septoria"),
    "tomato leaf bacterial spot": ("tomato", "bacterial"), "tomato leaf mosaic virus": ("tomato", "mosaic"),
    "tomato leaf yellow virus": ("tomato", "yellow"), "tomato mold leaf": ("tomato", "mold"),
    "tomato two spotted spider mites leaf": ("tomato", "spider"),
    "grape leaf": ("grape", "healthy"),          "grape leaf black rot": ("grape", "black_rot"),
}
norm = lambda s: s.lower().replace("_", " ").strip()

def our_class(crop, disease):
    hits = [i for i, c in enumerate(class_names)
            if crop in c.lower() and disease in c.lower().replace(" ", "_")]
    return hits[0] if len(hits) == 1 else None

X, Y, CROP = [], [], []
for split in ("train", "test"):
    root = f"/content/plantdoc/{split}"
    if not os.path.isdir(root): continue
    for folder in sorted(os.listdir(root)):
        key = norm(folder)
        target = our_class(*MAP[key]) if key in MAP else None
        if target is None:
            print("skipped (no matching class):", folder); continue
        for fn in os.listdir(os.path.join(root, folder)):
            try:
                img = load_img(os.path.join(root, folder, fn), target_size=(IMG_SIZE, IMG_SIZE))
            except Exception:
                continue   # PlantDoc has a few broken/non-image files
            X.append(tf.keras.applications.mobilenet_v2.preprocess_input(img_to_array(img)))
            Y.append(target)
            CROP.append(MAP[key][0])

X, Y = np.array(X), np.array(Y)
pred = np.argmax(model.predict(X, batch_size=64, verbose=0), axis=1)
exact = (pred == Y).mean()
crop_ok = np.mean([c in class_names[p].lower() for c, p in zip(CROP, pred)])
print(f"\nPlantDoc real-world test: {len(Y)} images")
print(f"  Exact disease correct: {100*exact:.1f}%")
print(f"  Correct crop (any disease): {100*crop_ok:.1f}%")
```
**Expect this to be much lower than the 95%+ validation number — possibly under 50%.** That is the point: lab-trained models are known to drop sharply on field photos. Report both numbers side by side; it shows you measured the real limitation instead of hiding it. Screenshot the output.

### Cell 10 — Test your own photo (with a confidence check)
```python
from google.colab import files
from tensorflow.keras.utils import load_img, img_to_array

THRESHOLD = 0.70
uploaded = files.upload()
for fn in uploaded:
    img = load_img(fn, target_size=(IMG_SIZE, IMG_SIZE))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(img_to_array(img))
    probs = model.predict(np.expand_dims(x, 0), verbose=0)[0]
    top = int(np.argmax(probs))
    if probs[top] < THRESHOLD:
        print(f"{fn} -> not sure (best guess {class_names[top]}, {100*probs[top]:.0f}%). Try a clearer, closer photo of one leaf.")
    else:
        print(f"{fn} -> {class_names[top]} ({100*probs[top]:.1f}% sure)")
```

### Cell 11 — Save and download
```python
model.save('phytocast_model_v2.keras')
with open('class_names_v2.txt', 'w') as f:
    f.write("\n".join(class_names))
files.download('phytocast_model_v2.keras')
files.download('class_names_v2.txt')
```
Put both files in the project's `Model/` folder, next to the v1 files (keep v1 as a fallback).

---

## Troubleshooting
- **Out of memory / session crashed** → Runtime → Restart session, set `BATCH = 16` in Cell 5, run from Cell 5 again (the download is kept).
- **Cell 5 finds 38 classes, not 39** → Cell 4's paths didn't match Cell 3's output; fix `TRAIN_DIR`/`VALID_DIR` and re-run Cells 4–5.
- **Disconnects mid-training** → keep the tab open and active. If it drops during Phase 2, Phase 1's result is still usable: skip to Cell 9.
- **Syngonium accuracy is low** → take more varied photos (especially close single leaves) and retrain.
