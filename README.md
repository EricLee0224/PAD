# PAD (NeurlPS'23 D&B Track, Submission): Official Project Repository.   
This repository provides the official PyTorch implementation code, data and models of the following paper:
> PAD: A Dataset and Benchmark for Pose-agnostic Anomaly Detection   
> [Qiang Zhou](https://scholar.google.com/citations?user=CMYTxUEAAAAJ&hl=en)(AIR), [Weize Li](https://ericlee0224.github.io/)(AIR), [Lihan Jiang](https://github.com/jianglh-WHU) (WHU), [Guoliang Wang](https://github.com/Cross-ZBuild) (AIR)   
> [Guyue Zhou](https://air.tsinghua.edu.cn/en/info/1046/1196.htm)(AIR), [Shanghang Zhang](https://www.shanghangzhang.com/)(PKU), [Hao Zhao](https://sites.google.com/view/fromandto)(AIR). <br>

> **Abstract:** 
*Object anomaly detection is an important problem in the field of machine vision and has seen remarkable progress recently. However, two significant challenges hinder its research and application. First, existing datasets lack comprehensive visual information from various pose angles. They usually have an unrealistic assumption that the anomaly-free training dataset is pose-aligned, and the testing samples have the same pose as the training data. However, in practice, anomaly can come from different poses and training and test samples may have different poses, calling for the study on pose-agnostic anomaly detection. Second, the absence of a consensus on experimental settings for pose-agnostic anomaly detection leads to unfair comparisons of different methods, hindering the research on pose-agnostic anomaly detection. To address these issues, we introduce Multi-pose Anomaly Detection (MAD) dataset and Pose-agnostic Anomaly Detection (PAAD) benchmark, which takes the first step to address the pose-agnostic anomaly detection problem. Specifically, we build MAD using 20 complex-shaped LEGO toys including 4k views with various poses, and high-quality and diverse 3D anomalies in both simulated and real environments. Additionally, we develop the PAAD framework, trained using MAD, specifically designed for pose-agnostic anomaly detection. Through comprehensive evaluations, we demonstrate the superiority of our dataset and framework. Furthermore, we provide an open-source benchmark library, including dataset and baseline methods which cover 8 anomaly detection paradigms, to facilitate future research and application in this domain.*<br>

<p align="center">
  <img src="assets/teaser(a).png" />
</p>

## Pose-agnostic Anomaly Detection Setting



## MAD: Multi-pose Anomaly Detection Dataset.

### MAD-Simulated Set

#### Data Directory
```
MAD-Sim
 └ 01Gorilla
   └ train
     └ good
       └ 0.png
       └ 1.png
   └ test  
     └ Burrs
     └ Missing
       └ 0.png
     └ Stains
       └ 0.png
     └ good
       └ 0.png
   └ ground_truth
   
   
   └ transforms.json
   
   
   
     └ train
     └ val
     └ test
 └ gtFine_trainvaltest
   └ gtFine
     └ train
     └ val
     └ test
```

### MAD-Real Set

#### Data Directory
```
Fishyscapes (OoD Dataset)
 └ leftImg8bit_trainvaltest
   └ leftImg8bit
     └ val
 └ gtFine_trainvaltest
   └ gtFine
     └ val
```
