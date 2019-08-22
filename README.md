# TextWorld Alpha

## Overview
This repository holds the final project deliverable for CSC 490, directed studies. It is an implementation of **Learning Latent Dynamics for Planning from Pixels** by *Danijar Hafner et al.* and **Sequence to Sequence Learning with Neural Networks** by *Ilya Sutskever et al.*

This implementation uses the 2018 Textworld API to generate stochastic environments for text-based training.

The paper is in the repository and called "Textworld_PlaNet.pdf".

## Get Started

The following information is needed in order to get this project running on your system.

### Environment

1. Create a `virtualenv` using `python3 -m virtualenv env`. Run `source env/bin/activate` to start the environment, and `deactivate` to close it.
2. Install dependencies using `pip3 install -r requirements.txt` both in the main and /PlaNet folders

### Train

Run `python3 custom_agent.py` to train on Textworld with the Seq2Seq model.

Run `python3 PlaNet/main.py` to train a tf.keras PlaNet implementation.

Note that the PlaNet model is not training properly yet, and fails at the 2nd loss function.
