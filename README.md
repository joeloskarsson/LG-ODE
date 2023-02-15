# Modified version of LG-ODE implementation
This repository contains a modified version of the LG-ODE implementation, used in our paper *Temporal Graph Neural Networks for Irregular Data*.
Please see our [main repository](https://github.com/joeloskarsson/tgnn4i) for the paper for more information.
We are grateful to the original LG-ODE authors for making their code [openly available](https://github.com/ZijieH/LG-ODE).

The modifications made to the implementation are mainly:

* Ability to load our datasets
* The use of a validation set during training
* Addition of a function to separately evaluate the model on test data

### Using the LG-ODE model with datasets from TGNN4I
To train and evaluate the LG-ODE model with our datasets:

* Convert the dataset to a suitable format for LG-ODE using the script `convert_to_lgode.py` found in the [main repository](https://github.com/joeloskarsson/tgnn4i).
* Place the converted dataset (the full folder) in the `data` directory in this repository.
* Train the LG-ODE model using `run_models.py`.
* To evaluate a trained model on the test set, use `run_models.py` with the flags `--test 1` and `--load` set to the name of the file to load model parameters from.
The evaluation script will then create a `.pt`-file with predictions.
* Use the script `eval_saved_predictions.py` in the [main repository](https://github.com/joeloskarsson/tgnn4i) to compute the final test errors.

# (Original README) LG-ODE

LG-ODE is an overall framework for learning continuous multi-agent system dynamics from irregularly-sampled partial observations considering graph structure.

You can see our Neurips 2020 paper [“**Learning Continuous System Dynamics from Irregularly-Sampled Partial Observations**”](https://arxiv.org/abs/2011.03880) for more details.

This implementation of LG-ODE is based on [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric) API.
## Data Generation

Generate simulated datasets (spring, charged particles) by running:

```bash
cd data
python generate_dataset.py
```

This generates the springs dataset, use `--simulation charged` for charged particles.

As simulated data is too large, we provide a toy-data from spring dataset and can be found under `data/example_data`

Motion dataset can be downloaded [CMU MoCap](http://mocap.cs.cmu.edu/)



## Setup

This implementation is based on pytorch_geometric. To run the code, you need the following dependencies:

* [Python 3.6.10](https://www.python.org/)

- [Pytorch 1.4.0](https://pytorch.org/)

- [pytorch_geometric 1.4.3](https://pytorch-geometric.readthedocs.io/)

  - torch-cluster==1.5.3
  - torch-scatter==2.0.4
  - torch-sparse==0.6.1

- [torchdiffeq 0.0.1](https://github.com/rtqichen/torchdiffeq)

- [numpy 1.16.1](https://numpy.org/)


## Usage
Execute the following scripts to train on the sampled data from spring system:

```bash
python run_models.py
```

There are some key options of this scrips:

- `--sample-percent-train`: This is the observed percentage in your training data.

- `--sample-percent-test`: This is the observed percentage in your testing data.

- `--solver` : This is for choosing your ODE Solver.

- `--extrap`: Set True to run in the extrapolation mode, otherwise run in the interpolation mode.


The details of other optional hyperparameters can be found in run_models.py.
### Citation

Please consider citing the following paper when using our code for your application.

```bibtex
@inproceedings{LG-ODE,
  title={Learning Continuous System Dynamics from Irregularly-Sampled Partial Observations},
  author={Zijie Huang and Yizhou Sun and Wei Wang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```
