# Interaction-Aware Adaptive Network for Drug-Drug Interaction Prediction

# Abstract

The prediction of drug-drug interactions (DDI) is crucial for drug safety and combination therapies. However, existing computational approaches face significant challenges in modeling drug interactions and effectively integrating multi-view information. To this end, AMIE-DDI, an Adaptive Multi-view Integration framework is proposed. First, Interaction-Enhanced Graph Transformer is designed to model complex relationships between drugs and capture the underlying interaction mechanisms. Second, a Multi-Channel Adaptive Fusion Module (MAF) is introduced to dynamically integrate information from different representations, enhancing feature learning and ensuring efficient multi-view feature integration. Finally, a Dynamic Interaction Scaling Prediction Module (DIS) is developed to adaptively adjust interaction intensity, thus improving both predictive accuracy and stability. Experimental results on multiple datasets demonstrate that AMIE-DDI outperforms state-of-the-art baselines in both warm-start and cold-start scenarios. Moreover, ablation studies and visualization experiments validate its capability to capture key motifs and enhance DDI prediction accuracy.


# 1. Requirements

To reproduce **AMIE-DDI**, the python==3.7,tensorflow-gpu==2.5.0, rdkit-pypi==2022.3.2 are required.

Of course, you can create your environment by env.yaml:
```sh
    $ conda env create -f env.yaml
```

# 2. Usage

### 2.1. Data

Data for AMIE-DDI can be downloaded from [ZhongDDI](https://github.com/LabWeng/MeTDDI/tree/main) and [ZhangDDI](https://github.com/zw9977129/drug-drug-interaction/tree/master/dataset).

### 2.2. Weight 
Weights for MeTDDI can be downloaded from [here](https://pan.baidu.com/s/1-PDKToc8Lf9xXTRgZGeDFQ?pwd=0000).

### 2.3. Useage 
For training:
```sh
    $ python train.py
```
For evaluating:

```sh
    $ python evaluate.py
```

# 3. Concat
Thank you for your interest in our work!

Please feel free to ask about any questions about the algorithms, codes, as well as problems encountered in running them so that we can make it clearer and better. You can either create an issue in the github repo or contact us at niudongjiang@qdu.edu.cn.
