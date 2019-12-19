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

### Clustering

DBSAN was used. Search for it inside ``analysis.ipynb`` for a raw usage.

### Training stability

Similar to dtw computation, use script ``batch_similarity.py`` that takes location for all logs where inside it searches for ``heatmaps\dtw_raw.plk``. At the same level as log dir, it create files ``cosine_similary.plk`` that stores raw cosine similary between all entries.
