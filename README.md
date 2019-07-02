# STAR-GCN
The MXNet implementation of [STAR-GCN: Stacked and Reconstructed Graph Convolutional Networks for Recommender Systems](https://arxiv.org/pdf/1905.13129.pdf)

## Prerequisite

### Requirements
  * [MXNet](https://mxnet.incubator.apache.org/) 
    * Please [build from source](https://mxnet.incubator.apache.org/versions/master/install/ubuntu_setup.html#build-mxnet-from-source)
    * Compile the customized kernel according to the [README](https://github.com/jennyzhang0215/STAR-GCN/blob/master/seg_ops_cuda/README.md)
  * GraphSampler
    * Compile the graphsampler API according to the [README](https://github.com/jennyzhang0215/STAR-GCN/blob/master/GraphSampler/README.md)
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
```bash
python setup.py develop
```

## Training
