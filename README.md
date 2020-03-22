# STAR-GCN
The MXNet implementation of [STAR-GCN: Stacked and Reconstructed Graph Convolutional Networks for Recommender Systems](https://www.ijcai.org/proceedings/2019/0592.pdf) in IJCAI 2019

## Prerequisite

### Requirements
  * [MXNet](https://mxnet.incubator.apache.org/) Version 1.5.0
    * Please [build from source](https://mxnet.incubator.apache.org/versions/master/install/ubuntu_setup.html#build-mxnet-from-source)
    * Compile the customized segment operator kernel according to [README](https://github.com/jennyzhang0215/STAR-GCN/blob/master/seg_ops_cuda)
  * GraphSampler
    * Compile the graphsampler API according to [README](https://github.com/jennyzhang0215/STAR-GCN/blob/master/GraphSampler)
  * [GluonNLP](https://gluon-nlp.mxnet.io/)
```bash
pip install gluonnlp
```

  * [spaCy](https://spacy.io/usage)
 ```bash
 pip install spacy
 python -m spacy download en
 ```

### Installation
Install the mxgraph python package
```bash
python setup.py develop
```

## Training
The training scripts are in the experiments directory. The chosen hyperparameters are stored in yml files in the cfg dirctory. 

For example, to train the MovieLens-100k in the transductive setting, we can run 
```bash
cd experiments/cfg
python ../STAR-GCN.py --ctx gpu0 --cfg transductive_ml-100k.yml
```


## Cite
Please cite our paper if you use this code in your own work:
```
@inproceedings{zhang2019star,
  title     = {STAR-GCN: Stacked and Reconstructed Graph Convolutional Networks for Recommender Systems},
  author    = {Zhang, Jiani and Shi, Xingjian and Zhao, Shenglin and King, Irwin},
  booktitle = {The 28th International Joint Conference on Artificial Intelligence},
  pages     = {4264--4270},
  year      = {2019}
}
```
