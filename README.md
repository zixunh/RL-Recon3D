# A Robot Learning Benchmark: Autonomous Construction of Digital Twin Tracking Dataset.
This resource was started as a final project for UC Berkeley's Deep Reinforcement Learning course [CS 285](https://rail.eecs.berkeley.edu/deeprlcourse/). I created a RLBench-based pipeline for robot learning on 3d object reconstruction trajectory finding. You can start with the following materials:

### Dependency
Version 4.1 of CoppeliaSim is required. Download:
- [Ubuntu 16.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu16_04.tar.xz)
- [Ubuntu 18.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz)
- [Ubuntu 20.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)

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

### Trajectory Finding
This repository is the implementation code focusing on 3D Object Reconstruction **Trajectory Finding**. I provided a tutorial that includes a mathematical proof to help fellow students grasp the fundamentals of 3D Reconstruction as well. Once you have obtained the trajectory using this repository, you can then refer to the tutorial [here](https://github.com/zixunh/RL-Recon3D/tree/main/recon) for guidance on performing 3D Reconstruction.

#### Imitation Learning
```
import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import ReachTarget

# To use 'saved' demos, set the path below
DATASET = 'PATH/TO/YOUR/DATASET'

action_mode = MoveArmThenGripper(
  arm_action_mode=JointVelocity(),
  gripper_action_mode=Discrete()
)
env = Environment(action_mode, DATASET)
env.launch()

task = env.get_task(ReachTarget)

demos = task.get_demos(2)  # -> List[List[Observation]]
demos = np.array(demos).flatten()

batch = np.random.choice(demos, replace=False)
batch_images = [obs.left_shoulder_rgb for obs in batch]
predicted_actions = predict_action(batch_images)
ground_truth_actions = [obs.joint_velocities for obs in batch]
loss = behaviour_cloning_loss(ground_truth_actions, predicted_actions)
```
####


### Contributing
Downstream tasks using our task building tool, such as building a 3D reconstruction dataset or a digital-twin tracking dataset, in addition to bug fixes, are very welcome! When building your task, please ensure that you run the task validator in the task building tool.

A full contribution guide is coming soon!

### Acknowledgements
Our Work is built upon [RLBench](https://github.com/stepjam/RLBench). Models were supplied from turbosquid.com, cgtrader.com, free3d.com, thingiverse.com, and cadnav.com. The digital twin tracking dataset that I used for imitation learning were offered by [OnePose](https://github.com/zju3dv/OnePose) and [DTTD2](https://github.com/augcog/DTTD2).

### Relevant Citation
```
@misc{DTTDv2,
    title={Robust Digital-Twin Localization via An RGBD-based Transformer Network and A Comprehensive Evaluation on a Mobile Dataset}, 
    author={Zixun Huang and Keling Yao and Seth Z. Zhao and Chuanyu Pan and Tianjian Xu and Weiyu Feng and Allen Y. Yang},
    year={2023},
    eprint={2309.13570},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@article{james2019rlbench,
    title={RLBench: The Robot Learning Benchmark \& Learning Environment},
    author={James, Stephen and Ma, Zicong and Rovick Arrojo, David and Davison, Andrew J.},
    journal={IEEE Robotics and Automation Letters},
    year={2020}
}

@article{sun2022onepose,
    title={{OnePose}: One-Shot Object Pose Estimation without {CAD} Models},
    author = {Sun, Jiaming and Wang, Zihao and Zhang, Siyu and He, Xingyi and Zhao, Hongcheng and Zhang, Guofeng and Zhou, Xiaowei},
    journal={CVPR},
    year={2022},
}
```
