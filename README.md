# An Interpretable Predictive Model for Early Detection of Hardware Failure

Link to the PDF will be added when available. Please ping me if I forget.

Enable tensorboard callback (see in `..."type": "TensorBoard"...` in configs for examples) to able view life all metrics.

## Data

  Fully pre-processed dataset is available in `data` directory. We reported `npy` instead of `hdf` due to compatibility on AC922 system. You can view some of the steps that have been done on normalized dataset at `data_masking` notebook.

## How to run

If you are using any of job scrips defined in `job_scripts` directory, move then into root directory

``python3.6 train.py -c config/config.json``

Some examples of config can be find in `config` directory

## Modify config

* `dataloader` consist of description of dataset.
  * `type` defines class name
  * `dir` defines directory inside the hdf5 file. If not specified then uses top level.
* `arch` specifies model
* `callbacks` is a list of functions (that are part of keras or custom) that are executed at different stage of traning.

The rest of parameters should be clear from their name.

### Submit job in hal

Batch of 6 training has been used but it can be generalized to any size. Create a config for each traning and submit simular script to below in HAL:

```ls -1 train_script_* | cut -d"_" -f3,4 | cut -d"." -f1 | xargs -I{} bash training_template.swb {}```

where it interprets each ``train_script_%d_%d.sh`` as a training script than then provided in default job parameters at ``training_template.swb``.

It automatically redirect hal's log stderr and stdout to ``logs/hal/retain/%d_%d.{out, err}``

## Result analysis

### DTW computation

At the end of training, if number of heatmaps was set to "-1", it will compute heatmap for a whole test set and store raw numpy arrays inside ``heatmaps`` folder i log dir of a current training.

``compute_dtw.py`` includes functionality to compte dtw for a given raw data in pickle. Result is stored in the same directory as raw data with name ``dtw_raw.pkl``. Second argument you can provide is number of threads to use for computation.

HAL system was used to compute dtw for a batch of models. To submit a batch of jobs:

```cat paths.txt | xargs -I{} bash job_dtw_template.swb {}```

where ``paths.txt`` has a line separted path to each raw heatmaps

### Training stability

Similar to dtw computation, use script ``batch_similarity.py`` that takes location for all logs where inside it searches for ``heatmaps/raw.plk``. At the same level as log dir, it create files ``cosine_similary.pkl`` that stores raw cosine similary between all entries.

Usage:

```bash job_similarity_template.swb "logs/fit/arch_name"```
