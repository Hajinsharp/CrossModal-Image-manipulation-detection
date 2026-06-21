# 🕵️‍♂️ Real-Time Multi-Modal Image Manipulation Detection

**A Thesis Project on Progressive Deep Learning Forensics for Image Tampering and AI Generation Detection**

This repository contains the code and methodology for a progressive ablation study on image manipulation detection. It explores the journey from traditional statistical forensics to a novel **Cross-Modal Transformer Fusion** architecture capable of cross-referencing multiple forensic signals in real-time. 

Finally, the project extends its capabilities via transfer learning to a **3-class detection system** (Authentic vs. Traditionally Manipulated vs. AI-Generated), bridging the gap between classic digital forensics and modern Deepfake detection.

---

## ⭐ Distinct Novelty: What & How

Most existing multi-modal forensic systems treat different image views as isolated signals that are simply concatenated at the end of a network. This project introduces two distinct architectural novelties:

### 1. ELA as a Learned Visual Stream (Novelty A)
* **What it is:** Instead of relying on Random Forests to evaluate hand-crafted statistical features (e.g., mean error, edge ratios) from Error Level Analysis (ELA), ELA is introduced as a complete visual tensor.
* **How it works:** The ELA pixel differences are normalized into a 3-channel visual input and fed directly into its own ImageNet-pretrained `EfficientNet-B0` backbone. The network learns *end-to-end* which specific ELA spatial patterns correlate with manipulation, rather than relying on human-defined heuristics.

### 2. Cross-Modal Transformer Fusion (Novelty B)
* **What it is:** Replacing standard scalar attention or late-stage concatenation with a **Modality Transformer Encoder**. 
* **How it works:** The features from all 5 streams `(B, 1280)` are stacked into a sequence `(B, 5, 1280)`. A Transformer Encoder (2 layers, 8 heads) is applied *over the modality dimension*. Each forensic stream attends to all others via multi-head self-attention. This enables complex inter-modality reasoning—for example, if the ELA stream detects heavy re-compression, the Transformer can dynamically upweight the DCT stream to confirm grid anomalies, while suppressing noisy RGB features.

---

## 🏛️ System Architecture

The diagram below outlines the primary contribution of this thesis: **CrossModalFusionNet**. 

```mermaid
graph TD
    Input["Input Image <br> JPEG / PNG / WebP"] --> S1["RGB Stream"]
    Input --> S2["ELA Stream ⭐ Novel"]
    Input --> S3["Noise Residual"]
    Input --> S4["DCT Log-Mag"]
    Input --> S5["ViT Global"]

    S1 --> P1["Resize 224x224 <br> Normalize"]
    S2 --> P2["JPEG re-save q=95 <br> Pixel Diff"]
    S3 --> P3["Gaussian Blur <br> Subtraction"]
    S4 --> P4["log(abs(DCT)) <br> Normalize"]
    S5 --> P5["Resize 224x224 <br> Patch Embed"]

    P1 --> B1["EfficientNet-B0 <br> freeze_until=4"]
    P2 --> B2["EfficientNet-B0 <br> freeze_until=4"]
    P3 --> B3["EfficientNet-B0 <br> freeze_until=4"]
    P4 --> B4["EfficientNet-B0 <br> freeze_until=4"]
    P5 --> B5["ViT-B/16 <br> CLS Token"]

    B1 --> F1("(B, 1280)")
    B2 --> F2("(B, 1280)")
    B3 --> F3("(B, 1280)")
    B4 --> F4("(B, 1280)")
    B5 --> Proj["Linear Proj"] --> F5("(B, 1280)")

    F1 --> Stack
    F2 --> Stack
    F3 --> Stack
    F4 --> Stack
    F5 --> Stack

    Stack{"Stack to sequence <br> B, 5, 1280"} --> Trans["⭐ Modality Transformer Encoder <br> 2L, 8H, cross-modal attention"]
    
    Trans --> Pool["Mean Pool <br> B, 1280"]
    Pool --> Head["Classifier Head <br> Linear -> ReLU -> Dropout -> Linear"]
    Head --> Out["Softmax Output <br> Authentic / Manipulated / AI-Generated"]
    
    classDef novel fill:#d5e8d4,stroke:#82b366,stroke-width:2px,color:#000;
    class S2,Trans novel;
