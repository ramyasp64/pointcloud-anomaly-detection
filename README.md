# 🔬 3D Point Cloud Anomaly Detection & Registration

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Open3D-3D%20Vision-darkgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Accuracy-93--95%25-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/RMSE-0.38mm-blue?style=for-the-badge"/>
</p>

> **R&D Research Lab Project** · Universität Koblenz · Mar 2026  
> *Fully automated, deterministic surface defect detection from 3D scan data.*

---

## 🎯 Project Overview

This project implements a robust, fully automated pipeline for **aligning 3D scan data (point clouds) with CAD reference models (STL meshes)** and classifying surface anomalies, defects, deformations, and background noise, with no manual intervention after initial setup.

The system uses a **"Cluster-First" anomaly detection approach**: instead of classifying individual noisy points, it identifies spatially coherent *clusters* of out-of-surface points and evaluates them against physical criteria (size, deviation, spatial extent, consistency). This eliminates false positives from scan edge artefacts while reliably detecting real structural defects.

**Results:** 93–95% classification accuracy across 6 independent industrial datasets, with RMSE as low as 0.38 mm.

---

## ⚙️ Pipeline Stages

```
┌─────────────────────────────────────────────────────────────┐
│                     PIPELINE WORKFLOW                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. DATA LOADING & SAMPLING                                  │
│     STL mesh + CSV point cloud → 30,000 sampled points       │
│     Cached to .npy for fast re-runs                          │
│                                                              │
│  2. MULTI-STAGE REGISTRATION                                 │
│     RANSAC Global Registration → rough 6-DOF alignment       │
│     Point-to-Plane ICP Refinement → sub-millimetre accuracy  │
│                                                              │
│  3. METADATA CACHING (OPTIMISATION)                          │
│     4×4 transformation matrix → registration_metadata.json  │
│     Subsequent runs: skip registration, load matrix directly │
│     Result: near-instant, 100% deterministic re-runs         │
│                                                              │
│  4. FLOOR & PLATFORM FILTERING                               │
│     Z-distribution analysis → bottom 5% flagged as table    │
│     Platform points → Class 3 (Background), not defects      │
│                                                              │
│  5. ANOMALY DETECTION, CLUSTER-FIRST                        │
│     Segmentation: points beyond Normal Threshold             │
│     DBSCAN Clustering → spatially coherent groups            │
│     Physical criteria evaluation:                            │
│       ✓ Size: ≥ 50 points in cluster                         │
│       ✓ Distance: average deviation significant?             │
│       ✓ Extent: physical volume/length present?              │
│       ✓ Consistency: median deviation high?                  │
│     → DEFECT (Class 2) or reclassified NORMAL (Class 1)      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎨 Output Visualisation

When the pipeline completes, an interactive **Plotly 3D viewer** opens for each dataset:

| Colour | Class | Meaning |
|--------|-------|---------|
| 🟢 **GREEN** | Class 1, Normal | Points matching the CAD surface |
| 🔴 **RED** | Class 2, Defect | Dense clusters deviating from model |
| 🔵 **BLUE** | Class 3, Background | Table surface, floor, noise outside footprint |

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Classification accuracy | **93–95%** across 6 datasets |
| Alignment RMSE | As low as **0.38 mm** |
| Datasets processed | 6 independent industrial gear scans |
| False positive approach | Cluster-first eliminates edge artefacts |

---

## 📁 Input & Output Files

**Input (per dataset):**
- `gear<N>.stl`, Reference CAD mesh
- `data<N>.csv`, Scanned point cloud (X, Y, Z columns)

**Output (per dataset):**
- `classified_data<N>.csv`, `[X, Y, Z, Class]`, main result
- `registration_metadata<N>.json`, transformation matrix, RMSE, thresholds
- `registration_transformation<N>.txt`, 4×4 refined transformation matrix
- `gear_registered<N>.ply`, Aligned mesh in scan coordinate space

---

## 🚀 Usage

```bash
# Clone and install
git clone https://github.com/ramyasp64/pointcloud-anomaly-detection.git
cd pointcloud-anomaly-detection
pip install -r requirements.txt

# Process all datasets in directory (auto-discovers gear<N>.stl + data<N>.csv pairs)
python optimized_registration.py
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| **Open3D** | 3D data processing, registration, RANSAC, ICP |
| **NumPy** | Numerical operations, matrix math |
| **SciPy** | ConvexHull, spatial structures |
| **Matplotlib** | 2D footprint path calculations |
| **Plotly** | Interactive 3D visualisation |

---

## 🔗 Connection to Healthcare AI

The anomaly detection methodology developed here (clustering-based, physically-motivated, threshold-calibrated detection) directly informs MediSense, which applies analogous anomaly scoring approaches to physiological signal data.

**Industrial quality control into Healthcare AI.** The same rigour, different domain.

---

## 📄 Project Documentation

- `Machine_Learning_Based_Anomaly_Detection_in_Point_Cloud_Data.pptx`, Full presentation
- `project_goals.md`, Original lab brief
- `gitlab_issue_description.md`, Task specification

---

## 👩‍💻 Author

**Ramya Subramanian Porselva Bharathi**  
M.S. Web and Data Science · Universität Koblenz, Germany  
[LinkedIn](https://www.linkedin.com/in/ramya_sp) · [GitHub](https://github.com/ramyasp64)
