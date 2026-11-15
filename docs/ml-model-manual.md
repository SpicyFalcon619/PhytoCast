# PhytoCast — Plant Disease Model Training Manual

**For:** the teammate training the AI model (Feature M1 — detection part)
**You need:** a laptop, a web browser, and a free Google account. **No hardware, no components.** Everything runs in the cloud (Google Colab), free.
**Time:** ~1 hour the first time (most of it is the computer training on its own).

---

## 1. What you're actually building (plain English)

You're teaching a computer to look at a photo of a plant leaf and say what disease it has (e.g., *"Tomato — Early Blight"*). You won't build the AI from scratch — you'll take a model that **already knows how to see images** and just teach it the last bit: recognizing plant diseases. This shortcut is called **transfer learning**, and it's why this only takes minutes instead of weeks.

You'll use a free, public dataset called **PlantVillage** (tens of thousands of labeled leaf photos), so you don't collect any images yourself.

**Honest scope (important — say this if asked):** the model learns from clean, close-up leaf photos on plain backgrounds. So it works great on *similar* clear photos, but less well on messy real-field pictures. In our project this is the **backup** feature (the main feature is disease *forecasting*), so that's totally fine — just demo it with clear leaf images.

---

## 2. Quick glossary (so the code makes sense)

- **Colab** = a free website that runs Python code in your browser, and gives you a free high-speed graphics chip (GPU) to train on.
- **Dataset** = the collection of labeled example images (PlantVillage).
- **Model** = the "brain" that learns. Ours is called **MobileNetV2** (small and fast).
- **Transfer learning** = starting from a model that already understands images, and fine-tuning it for our task.
- **Epoch** = one full pass through all the training images. More epochs = more learning (up to a point).
- **Accuracy** = the % of images the model labels correctly.

---

## 3. Step-by-step

### Step 1 — Open Colab and turn on the free GPU
1. Go to **[colab.research.google.com](https://colab.research.google.com)** and sign in with a Google account.
2. Click **File → New notebook**.
3. Turn on the free GPU: **Runtime → Change runtime type → Hardware accelerator → GPU (T4) → Save.**

You'll now type code into "cells" and run each with the **▶ play button** (or Shift+Enter). Copy each cell below **one at a time**, run it, wait for it to finish, then do the next.

---

### Cell 1 — Check everything is ready
```python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))
```
✅ **Expect:** a version number and a line showing a GPU device. If the GPU list is empty, redo the Runtime → GPU step above.

---

### Cell 2 — Get a free Kaggle token, then download the dataset

We get the PlantVillage photos from Kaggle rather than Google's dataset tool — it's more reliable. One-time setup first:

1. Go to **kaggle.com** and sign in (a free account is fine).
2. Click your profile picture (top-right) → **Settings**.
3. Scroll to **API** → click **Create New Token**. You'll see an **API Token** starting with `KGAT_...` — copy it. (Keep it private — it's tied to your account. You can always generate a new one later to replace it.)

```python
KAGGLE_TOKEN = "paste your KGAT_... token here"   # <- put your real token between the quotes

import os
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
with open(os.path.expanduser("~/.kaggle/access_token"), "w") as f:
    f.write(KAGGLE_TOKEN)
os.chmod(os.path.expanduser("~/.kaggle/access_token"), 0o600)

!pip install -q kaggle
!kaggle datasets download -d abdallahalidev/plantvillage-dataset -p /content/plantvillage --unzip
```
✅ **Expect:** it authenticates using your token, then downloads and unzips (~2 GB, a few minutes). Let it finish.

⚠️ **If this specific step gives a 403 Forbidden error:** open the dataset's page on kaggle.com in your browser, click any "I Understand and Accept" rules banner if one appears, then re-run the cell.

---

### Cell 2b — Check the folder layout (quick, important)
```python
!find /content/plantvillage -maxdepth 3 -type d | sort
```
✅ **Expect:** a list of folder paths. Find the one whose *subfolders* are named like `Tomato___healthy`, `Tomato___Early_blight`, etc. — that's the folder with the actual images. **Copy that exact path** — you need it for the next cell. If it doesn't look like the path used below, send the team the printed list so it can be adjusted.

---

### Cell 3 — Load the images
```python
DATA_DIR = "/content/plantvillage/plantvillage dataset/color"  # adjust if Cell 2b showed a different path

IMG_SIZE = 224
BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE

ds_train = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=123,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH
)
ds_val = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=123,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH
)

class_names = ds_train.class_names
num_classes = len(class_names)
print("Number of disease classes:", num_classes)
print("Example classes:", class_names[:5])

def prep(image, label):
    return tf.keras.applications.mobilenet_v2.preprocess_input(image), label

ds_train = ds_train.map(prep, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
ds_val   = ds_val.map(prep, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
print("Images ready.")
```
✅ **Expect:** it finds the images in each class folder, prints ~38 classes and some names, then "Images ready." (Resizing and shuffling happen automatically here — nothing extra to do.)

---

### Cell 4 — Build the model (transfer learning)
```python
base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'      # the pre-trained "already knows images" part
)
base.trainable = False       # freeze it — we only train the new last layer

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print("Model built.")
```
✅ **Expect:** it downloads the pre-trained model (a few seconds) and prints "Model built."

---

### Cell 5 — Train it (the main event)
```python
history = model.fit(ds_train, validation_data=ds_val, epochs=5)
```
✅ **Expect:** 5 epochs run one after another. Each line shows `accuracy` and `val_accuracy` climbing. On the GPU this takes roughly **10–20 minutes total**. By the end, `val_accuracy` should be around **0.90–0.97** (90–97%). Let it run — you can watch the numbers rise.

---

### Cell 6 — See the learning curve
```python
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.legend(); plt.title('Model accuracy')
plt.show()
```
✅ **Expect:** a graph with two rising lines. **Screenshot this — it's great for the report.**

---

### Cell 7 — Quick test on some validation images
```python
import numpy as np
for images, labels in ds_val.take(1):
    preds = model.predict(images)
    for i in range(8):
        pred = class_names[np.argmax(preds[i])]
        true = class_names[labels[i].numpy()]
        mark = "OK" if pred == true else "X"
        print(f"[{mark}] predicted: {pred:35s} actual: {true}")
    break
```
✅ **Expect:** 8 lines, most marked `[OK]` — the model's predictions vs the real answers.

---

### Cell 8 — Test on your OWN leaf photo (optional but fun)
```python
from google.colab import files
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

uploaded = files.upload()   # a window opens — pick a clear leaf photo from your computer
for fn in uploaded:
    img = load_img(fn, target_size=(IMG_SIZE, IMG_SIZE))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(img_to_array(img))
    x = np.expand_dims(x, 0)
    pred = model.predict(x)
    print(f"{fn}  ->  {class_names[np.argmax(pred)]}  ({100*np.max(pred):.1f}% sure)")
```
✅ **Expect:** it names the disease and a confidence %. Use a **clear, close-up leaf photo** for best results (search "tomato early blight leaf" and download one to test).

---

### Cell 9 — Save the model and download it
```python
model.save('phytocast_model.keras')

with open('class_names.txt', 'w') as f:
    f.write("\n".join(class_names))

from google.colab import files
files.download('phytocast_model.keras')   # downloads to your computer
files.download('class_names.txt')
```
✅ **Expect:** two files download to your computer: **`phytocast_model.keras`** (the trained brain) and **`class_names.txt`** (the list of diseases it knows).

---

## 4. What to hand back to the team

Send these two files to the group:
1. **`phytocast_model.keras`** — the trained model.
2. **`class_names.txt`** — the disease names.

Plus, for the report:
3. **A screenshot of the accuracy graph** (Cell 6).
4. **The final `val_accuracy` number** (from Cell 5) — e.g., "our model reached 94% accuracy."

That's everything the team needs to plug the model into the laptop dashboard later.

---

## 5. Troubleshooting

- **"No GPU" / training is super slow** → Runtime → Change runtime type → GPU → Save, then re-run from Cell 1.
- **`403 Forbidden` when running the Kaggle download in Cell 2** → open the dataset's page on kaggle.com in your browser, accept any rules banner, then re-run the cell. If it still fails, generate a fresh token from Kaggle Settings → API and paste that one in instead.
- **Cell 3 says a folder/directory doesn't exist, or finds 0 images** → the `DATA_DIR` path doesn't match what's actually on disk. Re-run Cell 2b, copy the exact path it prints, and paste it into `DATA_DIR` in Cell 3.
- **"Session crashed / out of memory"** → Runtime → Restart session, then run cells again from the top. If it keeps happening, change `BATCH = 32` to `BATCH = 16` in Cell 3.
- **Download disconnects / dataset stuck** → just re-run Cell 2; Kaggle downloads generally resume.
- **The notebook disconnects if you leave it idle** → keep the tab active while training.

---

## 6. If you want to make it simpler/faster (optional)

The dataset has ~38 classes (many crops). If you'd rather focus on just **tomato** (fewer classes, faster, higher accuracy for a demo), tell the team and we'll give you a 3-line filter to add. Not required — the full version works fine.

---

*Any error you can't solve: screenshot the red text and send it to the team. Everything here is safe — you can't break anything, and you can always Restart runtime and start over.*
