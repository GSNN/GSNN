# Graph Stochastic Neural Networks for Semi-supervised Learning

Code implementation of the paper: Graph Stochastic Neural Networks for Semi-supervised Learning, which has been accepted by NeurIPS 2020.

## Requirements
* `python 3.6.7`
* `numpy 1.15.4`
* `scipy 1.1.0`
* `scikit-learn 0.20.2`
* `matplotlib 3.0.2`
* `torch 1.1.0`
* `tqdm 4.31.1`

## Hardware Configurations
All experiments are conducted on a server with the following configurations:
* Operating System: `CentOS Linux release 7.4.1708`
* CPU: `Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.20GHz`
* GPU: `GeForce GTX TITAN X`

## Run the code
 
To try our code, you can use the IPython notebook `demo1_standard.ipynb`, `demo2_scarcelabel.ipynb` and `demo3_attack.ipynb` for three different experimental scenarios.
* Standard Experimental Scenario: `demo1_standard.ipynb`
* Label-Scarce Scenario: `demo2_scarcelabel.ipynb`
* Adversarial Attack Scenario: `demo3_attack.ipynb`

## Datasets
In the folder `./data`, we provide the following datasets:

| Dataset    | #(Node)   | #(Edge)   | #(Feature)   |   #(Class) |
| :--------: | :-------: | :-------: | :----------: | :--------: |
| Cora       | 2,708     | 5,249     | 1,433        | 7          |
| Citeseer   | 2,110     | 3,757     | 3,703        | 6          |
| Pubmed     | 19,717    | 44,324    | 500          | 3          |

Besides, to evaluate the performance of GSNN in the presence of adversarial attacks, we also provide the following poisoned graphs for Cora generated by different attack methods in the folder `./data` (attack budget: 0.05). 


#### Poisoned Graph Generated by Meta-Self
`cora_0.05edges_Meta-Self.npy`
#### Poisoned Graph Generated by Meta-Train
`cora_0.05edges_Meta-Train.npy`
#### Poisoned Graph Generated by Min-Max Attack
`cora_0.05_minmax.npy`


