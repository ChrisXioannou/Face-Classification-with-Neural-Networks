# Face-Classification-with-Neural-Networks
Face Classification with Neural Networks
testing 123
![Network Diagram](https://github.com/ChrisXioannou/HomeLabDirectory/blob/main/assets/diagam.png)
---------------------------
Real vs. Fake Face Detection with Deep Learning

A clean, minimal repository README for a project that classifies face images as real or fake using deep learning. It covers a compact dataset experiment, fine-tuning of pretrained CNNs, a custom CNN baseline, and a concise statistical comparison of results.

🔍 Abstract

This study addresses binary classification of faces into real and fake (GAN-generated). The goal is to build and evaluate neural network models capable of distinguishing authentic from synthetic faces with reliable accuracy for use in security, social media integrity, and content moderation.

🧩 Problem Description

Detecting fake faces, often produced by GANs, is challenging and high-stakes. Robust classification helps reduce misinformation and abuse while supporting trust and safety systems.

🧪 Solution Strategy

Start with a small experimental subset of the dataset to validate pipelines and models.

Apply fine-tuning on pretrained CNNs and train a Custom CNN for comparison.

Use standard ML practice: preprocessing, EDA, k-fold validation, training, fine-tuning, evaluation, plus ANOVA for statistical significance across models.

🗂️ Data

Dataset: 140K Real and Fake Faces (Kaggle)
A small sample was used for the initial implementation:

Total images: 400

Class balance: 200 real, 200 fake

Splits:

Train: 320 images (80%)

Validation: 40 images (10%)

Test: 40 images (10%)

Note: Final training, fine-tuning, and evaluation in this repo focus on the small subset due to compute constraints.

Folder structure (example):

/content/smallSet/
├── real/
│   ├── image1.jpg
│   └── ...
└── fake/
    ├── image1.jpg
    └── ...

🛠️ Models

Five CNN approaches were evaluated:

Custom CNN

Xception

InceptionV3

DenseNet121

EfficientNetB7

Preprocessing & Input

Normalization: rescale=1./255

No augmentation for the small subset

Target sizes:

299×299 for InceptionV3 and Xception

224×224 for others

Batch size: 16 for smaller models, ~8 for EfficientNetB7

Data loaders

Either manual directory iteration (os) or Keras ImageDataGenerator.

📊 Results (Summary)

Best: InceptionV3 (fine-tuned) — Accuracy: 90%, Loss: 0.3041

Xception and DenseNet121 — both at 77.5% accuracy, with DenseNet121 showing the lower loss

EfficientNetB7 improved with tuning but did not surpass the above

Custom CNN underperformed relative to pretrained models

Statistical test (ANOVA) on model accuracies:

F-statistic: 236.4000

p-value: 0.0000
Conclusion: Performance differences are statistically significant.

Final choice: InceptionV3 (fine-tuned) as the recommended model for this task.

🚀 How to Use

Notebooks walk through training and evaluation on the small subset.

Update dataset paths in the notebook to match your environment.

Typical workflow: load data → choose model → train/fine-tune → evaluate → compare.

🔭 Future Work

Feature map visualizations to interpret salient regions

Expand dataset beyond the small subset

Data augmentation (rotations, flips, brightness) to reduce overfitting

More k-folds for robust estimates

Multimodal inputs (e.g., EXIF, video) for deepfake detection at scale

Real-time inference for browser/mobile deployment

Additional regularization to address observed overfitting

📚 References

Chollet, F. (2017). Xception: Deep Learning with Depthwise Separable Convolutions. https://arxiv.org/abs/1610.02357

Szegedy, C. et al. (2016). Rethinking the Inception Architecture for Computer Vision. https://arxiv.org/abs/1512.00567

Huang, G. et al. (2017). Densely Connected Convolutional Networks. https://arxiv.org/abs/1608.06993

Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for CNNs. https://arxiv.org/abs/1905.11946

Kaggle Dataset: 140K Real and Fake Faces. https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces

📦 Libraries

Google Colab, Google Drive, TensorFlow/Keras, PyTorch, OpenCV, NumPy, scikit-learn, Matplotlib, ZipFile, OS, random

🤝 Contribute & Support

If this project helps you, star the repo.
PRs, issues, and feature requests are welcome.
