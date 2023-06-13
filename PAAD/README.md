# PAAD: Pose-agnostic Anomaly Detection

## Installation

To start, I recommend to create an environment using conda:

```
conda create -n pad python=3.8
conda activate pad
```

Clone the repository and install dependencies:

```
git clone https://github.com/jianglh-WHU/PAAD.git
cd inerf
pip install -r requirements.txt
```

## How to use

## Train

First, you should download our MLAD-Sim dataset, just from [here](https://drive.google.com/file/d/1S1rYgPyxFjCuLf1Z-JLyfS2lka4_pykf/view) and put the downloaded folder in the "data/MLAD-Sim" folder

```
├── data 
│   ├── MLAD-Sim  
```

To run the algorithm on *9(Gorilla)* object

```python
python anomaly_nerf_lego.py --config configs/LEGO-3D/9.txt --class_name 9
```

All other parameters such as *batch size*, *class_name*, *dataset_type* you can adjust in corresponding config [files](https://github.com/jianglh-WHU/PAAD/tree/main/configs/LEGO-3D).

All NeRF models were trained using this code https://github.com/yenchenlin/nerf-pytorch/

You can use our ckpts on MLAD-Sim in [ckpts](https://github.com/jianglh-WHU/PAAD/tree/main/ckpts/LEGO-3D)

And iNeRF using the code https://github.com/salykovaa/inerf

## Evaluate

The test script requires the --obj arguments

```
python auroc_metric_feature.py --obj 9
```

# Examples

![data_9](imgs/data_9.png)

![data_15](imgs/data_15.png)

![data_25](imgs/data_25.png)

![data_59](imgs/data_59.png)

