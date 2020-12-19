
This repository mainly contains the C++ code used to conduct the gridworld experiments reported in the paper "Optimizing BROIL using Deep Bayesian method for Reward Learning" for the course CMSC818B-Decision Making For Robotics.
Note that this code repository has not been optimized for memory management or speed. 
  
  ### Getting started
*****
  - `git clone https://github.com/SmritiUMD/BROIL_test.git`
*****
  - `cd brex_gridworld_cpp`
  - `mkdir build`

- Build the experiment
```
make brex_gridworld_basic_exp_optsubopt 
```
- Run the experiment
```
./brex_gridworld_basic_exp_optsubopt
```

This will write the results to `./data/brex_gridworld_optsubopt/`
  
  Experiment will take around 10+ minutes to run since it runs 100 replicates for each number of demonstrations. 

  - Generate the data shown in paper
  ```
  python scripts/evaluatePolicyLossOptSubopt.py
  `
