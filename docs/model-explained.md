# PhytoCast — The Disease-Detection Model, Explained

A plain summary of what was built, how, and what the results mean — enough to present it and answer questions.

---

## 1. What the model does

It looks at a photo of a single leaf and names the crop and its disease (or "healthy") — for example, *Tomato — Early blight*. If it is less than 70% confident, it says "not sure" instead of guessing.

In PhytoCast this is the **secondary** feature. The primary one is the sensor-based forecast, which predicts disease risk from temperature and humidity *before* symptoms appear.

---

## 2. The data

| | |
|---|---|
| Dataset | New Plant Diseases Dataset (Kaggle) |
| Images | **87,867** — 70,295 for training, 17,572 for validation |
| Classes | **38** — 14 crops (tomato, potato, apple, corn, grape, …), each healthy and/or diseased |
| Origin | An augmented version of PlantVillage: the original lab photos plus rotated, flipped and re-lit copies |

**Real-world test set:** PlantDoc — **2,578** photos of leaves in real field conditions (cluttered backgrounds, natural light), collected from the web. The model never saw these during training.

---

## 3. How it was trained

**Transfer learning with MobileNetV2.**
MobileNetV2 is a neural network already trained on ImageNet (about 1.4 million general images), so it already knows how to recognise edges, textures and shapes. We kept that knowledge and taught it only the last step: plant diseases. This is why training took minutes, not weeks.

We chose MobileNetV2 because it is small and fast enough to run on an ordinary laptop.

**Two training phases:**
1. **Phase 1** — the pretrained network is frozen; only a new final layer (38 outputs) is trained. 3 epochs.
2. **Phase 2 (fine-tuning)** — the last 30 layers of MobileNetV2 are unfrozen and adjusted with a very small learning rate (0.00001), so the network adapts to leaves without losing what it already knows. 2 epochs.

**Techniques used:**
- **Data augmentation** — during training, images are randomly flipped, rotated and zoomed, so the model does not memorise exact photos.
- **Dropout (30%)** — randomly switches off parts of the network during training, to reduce overfitting.
- **Class weighting** — balances classes with fewer images.
- **Checkpointing** — the best version is saved to Google Drive after every epoch, so a Colab disconnect does not lose the work.

Tools: TensorFlow / Keras on Google Colab's free GPU (NVIDIA T4).

---

## 4. Results

| Test | Result |
|---|---|
| Validation, after Phase 1 | **92.4%** |
| Validation, after fine-tuning | **94.7%** |
| Real-world photos (PlantDoc) — exact disease | **20.0%** |
| Real-world photos (PlantDoc) — correct crop | **41.3%** |

For scale: with 38 classes, random guessing scores about 2.6%.

---

## 5. What the results mean

The model is accurate on images that look like its training data, and much weaker on real field photos. This drop is known as **domain shift**: the training photos are single leaves on plain backgrounds under even lighting, while field photos have clutter, shadows, several leaves and different angles.

This is the reason PhytoCast does not rely on the camera. Its main defence is the sensor-based forecast, which does not depend on image quality and acts before a disease is visible. Image detection is a secondary check.

We measured this gap on purpose, with a separate real-world test set, rather than reporting only the validation number.

---

## 6. What changed after the faculty's feedback

| | Before | After |
|---|---|---|
| Training images | 54,305 | **87,867** |
| Training | One phase | Two phases, with fine-tuning |
| Augmentation | None | Flip, rotate, zoom |
| Real-world test | None | PlantDoc, 2,578 field photos |
| Wrong confident guesses | Possible | Below 70% confidence it answers "not sure" |

---

## 7. Limitations

- **Crops only.** The model covers the 38 classes in the dataset. Our own plant, a Syngonium (arrowhead plant), is not one of them — no public dataset of diseased Syngonium exists. A Syngonium photo correctly returns "not sure".
- **The dataset is augmented.** The 87,867 images come from fewer original photos, plus modified copies of them.
- **Validation is optimistic.** The validation images come from the same source as the training images, so the validation score is higher than real-world performance. The PlantDoc result is the honest estimate.

---

## 8. One problem we found and fixed

The first attempt at the new model only reached **72%**. The cause was one augmentation step (random contrast). In the current version of Keras it clips pixel values to 0–255, but our images had already been scaled to −1…1, so it destroyed half of every image during training. Removing that step brought validation accuracy back above 90%.

---

## 9. Likely questions

**What is transfer learning?**
Reusing a network already trained on a large general dataset, and retraining only the final part for a new task. It needs far less data and time than training from scratch.

**Why MobileNetV2 and not a bigger model?**
It is small and fast, so it runs on a laptop in real time. Bigger models (such as ResNet-50) might score slightly higher, but are slower and heavier.

**Why is real-world accuracy so low?**
Domain shift — the model learned from clean lab photos. Real photos look different. This is a known, general problem in plant-disease detection, not a bug in our model.

**Why not train on PlantDoc as well?**
It has only about 2,600 images, and using it for testing gives an independent, honest measure of real-world performance. Training on it would remove that measure.

**How could it be improved?**
Train on more real field photos, especially of the plants actually being monitored; photograph our own Syngonium to add it as a class; try a larger model.

**Does it work on your plant?**
Not for disease — there is no data for Syngonium diseases. Our plant is protected by the sensor forecast, which uses Syngonium-specific temperature and humidity limits.
