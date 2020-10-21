# Soft Q-learning for Space Robots
 
Codes accompanying the paper:  
C. Yan, Q. Zhang, Z. Liu, X. Wang and B. Liang, "Control of Free-Floating Space Robots to Capture Targets UsingSoft Q-Learning," *2018 IEEE International Conference on Robotics and Biomimetics (ROBIO)*, Kuala Lumpur,Malaysia, 2018, pp. 654-660. (doi: 10.1109/ROBIO.2018.8665049)
 
Based on [Haarnoja](https://github.com/haarnoja/softqlearning), this framework provides an implementation of Soft Q-learning algorithm for controlling space robots to capture targets,
and supports running experiments on V-REP simulation environments.
 
## Getting Started
 
1. You will first need to clone [rllab](https://github.com/rll/rllab), and have its path added to your PYTHONPATH environment variable.

2. Install requirements: `pip install -r requirements.txt`

3. Install and run [V-REP](https://www.coppeliarobotics.com/downloads). Load the V-REP model.  
  Some example V-REP models are available in the `VREP MODEL` file.  
  Adding the flag `-h` to run V-REP in a headless mode.
