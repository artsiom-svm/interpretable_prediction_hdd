## How to run:
``python3.6 train.py -c config/config.json``

Some examples of config can be find in `config` directory

## Modify config

* `dataloader` consist of description of dataset.
  * `type` defines class name
  * `dir` defines directory inside the hdf5 file. If not specified then uses top level.
* `arch` specifies model
* `callbacks` is a list of functions (that are part of keras or custom) that are executed at different stage of traning.

The rest of parameters should be clear from their name.
