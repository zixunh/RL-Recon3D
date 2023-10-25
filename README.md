# Robot Learning Benchmark for 3D Object Reconstruction

### Dependency
Version 4.1 of CoppeliaSim is required. Download:
[Ubuntu 16.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu16_04.tar.xz)
[Ubuntu 18.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz)
[Ubuntu 20.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)

Add the following to your ~/.bashrc file: (NOTE: the 'EDIT ME' in the first line)
```
export COPPELIASIM_ROOT=EDIT/ME/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Once you have downloaded CoppeliaSim, you can pull PyRep from git:
```
git clone https://github.com/stepjam/PyRep.git
cd PyRep
pip3 install -r requirements.txt
pip3 install .
```

Then you can install RLBench directly via pip:
```
pip install git+https://github.com/stepjam/RLBench.git
```


### 