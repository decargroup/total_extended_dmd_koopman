# Asymptotically Stable Data-Driven Koopman Operator Approximation with Inputs using Total Extended DMD

Companion code for Asymptotically Stable Data-Driven Koopman Operator Approximation with Inputs using Total Extended DMD

## Installation

To clone the repository and its
[submodule](https://github.com/ramvasudevan/soft-robot-koopman), which contains
the soft robot dataset, run
```sh
$ git clone --recurse-submodules git@github.com:decargroup/forward_backward_koopman.git
```

To install all the required dependencies for this project, it is recommended to create a virtual environment. After activating the virtual environment, run
```sh
(venv) $ pip install -r ./requirements.txt
```

The LMI solver used, MOSEK, requires a license to use. You can request personal
academic license [here](https://www.mosek.com/products/academic-licenses/).

[^1]: On Windows, use `> \venv\Scripts\activate`.
[^2]: On Windows, place the license in `C:\Users\<USER>\mosek\mosek.lic`.

## Usage

### Generating data

To generate the data required for fitting the models, run
```sh
python preprocess.py --multirun preprocessing.data.noise=0,0.00001,0.0001,0.001,0.01,0.1,1 preprocessing=soft_robot,nl_msd
```

### Fitting models

To fit all models and obtain all predictions for the figures of the soft robot in the paper except for Figure 6, run
```sh
python main.py --multirun robot=soft_robot variance=0.01,0.1 regressors@pykoop_pipeline=EDMD,EDMD-AS,FBEDMD,FBEDMD-AS,TEDMD,TEDMD-AS lifting_functions@pykoop_pipeline=soft_robot_poly2_centers10
```

Similarly, for the Duffing oscillator, run
```sh
python main.py --multirun robot=nl_msd variance=0.01,0.1 regressors@pykoop_pipeline=EDMD,EDMD-AS,FBEDMD,FBEDMD-AS,TEDMD,TEDMD-AS lifting_functions@pykoop_pipeline=nl_msd_poly2_centers10
```

To generate FIgure 6, run separetly
```sh
python main.py --multirun regressors@pykoop_pipeline=EDMD-AS,FBEDMD-AS,TEDMD-AS variance=0,0.00001,0.0001,0.001,0.01,0.1,1 pred=false
```

### Generating plots

To generate all figures except Figure 6, run 
```sh
python plot.py --multirun what_to_plot.variance=0.01,0.1 plotting@what_to_plot=soft_robot_plots,nl_msd_plots
```

To generate Figure 6, run separately 
```sh
python plot.py plotting@what_to_plot=frob_err_plots
```

The plots will be found in `build/figures/paper`.