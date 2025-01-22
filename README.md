# Lightweight ResNet-Based Deep Learning for Photoplethysmography Signal Quality Assessment

---

### Overview

This repository contains the implementation of a lightweight ResNet-based deep learning model integrated with Squeeze-and-Excitation (SE) blocks. The model is optimized for photoplethysmography (PPG) signal quality assessment to classify signals into "good" or "bad" quality, enabling enhanced signal processing.

---

### Model Architecture

Below is the architecture of the lightweight ResNet model used for PPG signal quality assessment:

![Model Architecture](Fig.png)

*Figure 1: Overview of the proposed LRS-SE framework. The first panel illustrates the input signal format, comprising three channels: the raw PPG signal, the first derivative of PPG (FDP), and the second derivative of PPG (SDP), as well as the overall LRS-SE architecture for signal quality classification ('good' or 'bad'). The second panel details the LRS-SE structure with two Res blocks (Layer 1 and Layer 2). The third panel depicts the composition of a single Res block, which includes two Basic blocks. The fourth panel highlights the Basic block, integrating a Squeeze-and-Excitation (SE) block to implement channel attention.*


