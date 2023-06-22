# Constrained-HIDA
Official code for the ECML PKDD 2023 conference paper "Constrained-HIDA: Heterogeneous Image Domain Adaptation Guided by Constraints".

## Requirements:
* Tensorflow 1.15

## Data
Download data.zip file from the following [link](https://seafile.unistra.fr/f/90434a7cd054499883f5/?dl=1).

Put the zip file inside of the "src" folder and unpack it.

## Training constrained-HIDA
Here, an example of running a training of Constrained-HIDA. From "src" folder, run the following terminal command:
```
python run.py --usecase constrained_hida --num_constraints 160 --exp_name constrained_hida_160
```

