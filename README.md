## RCQE
# Automatic Code Quality Estimation in Multi-round Code Review

This repo provides the code for reproducing the experiments in the paper: Automatic Code Quality Estimation in Multi-round Code Review.

The dataset is available on Zenodo: [https://zenodo.org/records/11109683](https://zenodo.org/records/11109683)

## Dependency

```sh
conda install nltk
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install transformers
```
## Experiments

- 🌱 The code for the model RCQE-T5 family is in ./RCQE-T5, prompt-tuning by running ./RCQE-T5/run_train.sh.
- 💬The code for the model RCQE family is in ./RCQE, fine-tuned by running the shell script in ./code/sh.
- ⚡The code for baseline models is in ./SimAST-GCN-master, the original repo is: https://github.com/SimAST-GCN/SimAST-GCN.

