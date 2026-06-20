# 🔵 KNN Classifier Visualizer

> An interactive, visual machine-learning playground built with **React + TypeScript + Vite**.

[![Deploy to GitHub Pages](https://github.com/deepthi-tr05/KNN-Classifier-Visualizer/actions/workflows/deploy.yml/badge.svg)](https://github.com/deepthi-tr05/KNN-Classifier-Visualizer/actions/workflows/deploy.yml)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-%F0%9F%9A%80-brightgreen)](https://deepthi-tr05.github.io/KNN-Classifier-Visualizer/)

### 🚀 [Live Demo → https://deepthi-tr05.github.io/KNN-Classifier-Visualizer/](https://deepthi-tr05.github.io/KNN-Classifier-Visualizer/)

---

## 🎯 What is this?

This project is an **interactive visualizer** for two foundational machine-learning algorithms:

1. **K-Nearest Neighbours (KNN)** — a classic classification algorithm that predicts by majority vote among the K closest training points.
2. **FIND-S Algorithm** — a concept-learning algorithm that finds the most specific hypothesis consistent with all positive training examples.

No server needed — everything runs in the browser.

---

## ✨ Features

### 🔵 KNN Tab
| Feature | Description |
|---------|-------------|
| 🌸 Iris Dataset | Fisher's 150-sample, 4-feature classic dataset |
| 📉 PCA Projection | 4D data reduced to 2D via Jacobi eigenvalue method |
| 🎚️ Feature Sliders | Drag to set query point (sepal/petal length & width) |
| 🔢 K Slider | Adjust K from 1–15 and see neighbours highlight instantly |
| 🏷️ Live Prediction | Shows predicted class + each neighbour's Euclidean distance |

### 🧠 FIND-S Tab
| Feature | Description |
|---------|-------------|
| 📋 Step-by-step | Walk through each training example one by one |
| 🎨 Diff table | Updated attributes glow cyan; generalised `?` shown in orange |
| 📖 Enjoy Sport dataset | 4 examples, 6 attributes — the textbook FIND-S demo |
| 📌 Algorithm recap | Built-in explanation panel |

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | React 19 + TypeScript |
| Build Tool | Vite 7 |
| Styling | Tailwind CSS v4 + Vanilla CSS (glassmorphism) |
| Charts | Recharts |
| Math | Custom PCA (Jacobi eigenvalue) — zero external math deps |
| Deploy | GitHub Pages via GitHub Actions |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/deepthi-tr05/KNN-Classifier-Visualizer.git
cd KNN-Classifier-Visualizer

# 2. Install dependencies
npm install

# 3. Start the dev server
npm run dev
```

Open [http://localhost:5173/KNN-Classifier-Visualizer/](http://localhost:5173/KNN-Classifier-Visualizer/)

### Build for production
```bash
npm run build
# Output: dist/index.html (single self-contained file)
```

---

## 📁 Project Structure

```
KNN-Classifier-Visualizer/
├── .github/
│   └── workflows/
│       └── deploy.yml       # Auto-deploy to GitHub Pages on push
├── src/
│   ├── App.tsx              # Root component — KNN tab + FIND-S tab
│   ├── main.tsx             # React entry point
│   ├── index.css            # Global styles & design tokens
│   ├── data/
│   │   ├── irisData.ts      # Fisher's Iris dataset (150 × 4)
│   │   └── findsData.ts     # Enjoy Sport dataset + FIND-S logic
│   └── utils/
│       ├── pca.ts           # PCA via Jacobi eigenvalue decomposition
│       └── cn.ts            # Tailwind class merger utility
├── index.html
├── vite.config.ts
├── tsconfig.json
└── package.json
```

---

## 🧮 Algorithms Implemented

### K-Nearest Neighbours
```
For a query point q with k neighbours:
  1. Compute Euclidean distance to every training point
  2. Select k nearest neighbours
  3. Return majority class label among neighbours
```

### Principal Component Analysis (PCA)
```
Used to project 4D Iris features → 2D for visualisation:
  1. Centre data by subtracting column means
  2. Compute covariance matrix
  3. Extract eigenvectors via Jacobi iteration
  4. Project onto top-2 principal components
```

### FIND-S
```
Start with most specific hypothesis H = ∅
For each positive training example e:
  For each attribute i:
    If H[i] ≠ e[i]: generalise H[i] → '?'
Return H
```

---

## 📜 License

MIT © [Deepthi T R](https://github.com/deepthi-tr05)