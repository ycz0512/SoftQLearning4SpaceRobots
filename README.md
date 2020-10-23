# Soft Q-learning for Space Robots
 
Code accompanying the paper:  
C. Yan, Q. Zhang, Z. Liu, X. Wang and B. Liang, "Control of Free-Floating Space Robots to Capture Targets UsingSoft Q-Learning," *2018 IEEE International Conference on Robotics and Biomimetics (ROBIO)*, Kuala Lumpur,Malaysia, 2018, pp. 654-660. (doi: 10.1109/ROBIO.2018.8665049)
 
Based on [Haarnoja](https://github.com/haarnoja/softqlearning), this framework provides an implementation of Soft Q-learning algorithm for controlling space robots to capture targets,
and supports running experiments on V-REP simulation environments.
 
## Getting Started

1. You will first need to clone [rllab](https://github.com/rll/rllab), and have its path added to your PYTHONPATH environment variable.

2. Install requirements: `pip install -r requirements.txt`

3. Install and run [V-REP](https://www.coppeliarobotics.com/downloads). Load the V-REP model.

- Some example V-REP models(.ttt) are available in the `VREP MODEL` file.
- Adding the flag `-h` to run V-REP in a headless mode.

## Examples
**Note:** Before you start running any commands below, always make sure that the V-REP simulator is on (whether in headless mode or not), and the environment you have in the command should be the same as what you have in the V-REP simulator.

### Loading and Training Agents
To load a pre-trained model and train the agent, run:
```
python experiment/run.py --env=SpaceRobot3link --load_model=<model-directory>
```
- `SpaceRobot3link` can be replaced with `SpaceRobotDouble3link`, where `SpaceRobot3link` specifies the space robot with a single 3-DoF manipulator and `SpaceRobotDouble3link` specifies the space robot with dual 3-DoF manipulators. If you remove the flag, `--env` will be `SpaceRobot3link` by default.
- `<model-directory>` specifies the directory of `.ckpt` file that contains the pre-trained model. If you remove the flag, training will start from scratch.
- The log(.txt) and model(.ckpt) will be saved to the `../data` directory by default.

### Visualizing Agents
To simulate the agent, run:
```
python experiment/visualize.py --env=SpaceRobot3link --model=<model-directory>
```
- This will simulate the agent saved at `<model-directory>` (the directory of `.ckpt` file that contains the trained model).
- The log(.txt) files generated in the simulation will be saved to `../viz_data` directory by default.
