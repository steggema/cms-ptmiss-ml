# cms-ptmiss-ml

## Setup

Requires [miniconda (Python 3.7)](https://docs.conda.io/en/latest/miniconda.html#linux-installers). Install as follows:

```shell
curl -OL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

After installation, install the environment using `environment.yml` included in this repository:

```shell
conda env create --file=environment.yml
```

For recurrent setup, also do the following or add the equivalent lines in a `setup.sh` script:

```shell
conda init <shell>
```

Activate the environment before running:

```shell
conda activate tf_gpu
```

## Running

Example training command (mind that using inputs from eos can be very slow):

```shell
python -i wpt-pf-grav.py -i /eos/user/s/steggema/ptmiss/dy_pf_chunk56.h5 --embedding
```

In order to run over a large number of samples, mixing different processes, create a "virtual hdf5" index file, and then run it with the command above:

```shell
python create_virtual_hd5.py -i /tmp/${USER}/dy_pf_chunk60.h5,/tmp/${USER}/tt_pf_chunk240.h5,/tmp/${USER}/dy_pf_chunk59.h5,/tmp/${USER}/tt_pf_chunk249.h5 -o out_20190930_v1.h5 -d X,X_c_0,X_c_1,X_c_2,Y,Z
```
