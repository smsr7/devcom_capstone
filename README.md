# Operation Safe Passage (OSP)

**Operation Safe Passage (OSP)** is a Gymnasium environment designed to simulate the safe traversal of a minefield using autonomous agents. Within the simulation, agents include an **Unmanned Aerial Vehicle (UAV)**, an **Unmanned Ground Vehicle (UGV)**, and two types of mine detection methods—**AI-based and Human-based**.

The UAV scans potential traversal cells, gathering mine detection confidence scores from either AI or Human methods. Detection accuracy varies based on terrain metadata associated with the map. To ensure safe and trustworthy navigation, the system must effectively route the UAV, select appropriate scanning methods (AI or Human), and guide the UGV to the target location while minimizing encounters with mines.

---

## Installation

1. **Create the Conda Environment**

To build the Conda environment named `devcom`, run:
```
conda env create -f environment.yml
```
Activate the environment:
```
conda activate devcom
```

2. **Verify Installation and Generate Test Map**
Verify the successful installation by generating a test map:
```
python mapGenerator.py
```

## Running the Agent
Run the reinforcement learning agent with:
```
python RL_new.py
```

### Select desired experimental configurations by modifying the main loop in `RL_new.py`:

- **`main()` function:**  
  Routes the UAV to the goal, and upon reaching the goal, navigates the UGV.

- **`train()` function:**  
  Trains or deploys the reinforcement learning agent.

- **`ugv()` function:**  
  Directly routes the UGV to the goal.


### Desired experimental configurations for the Two UAV version in `RL_new_new_two.py`:
- **`main()` function:**  
  Routes the UAV to the goal, and upon reaching the goal, navigates the UGV.

- **`train()` function:**  
  Trains or deploys the reinforcement learning agent.

- **`ugv()` function:**  
  Directly routes the UGV to the goal.

## Results
Following the completion of training, or routing methods, a `out.csv` file is created containing the individual moves of the UGV, UAV, and action. An example of this data is below:

`action,start_uav,start_ugv,start_scanner,end_uav,end_ugv,end_scanner`

`"[4, 1, 5]","(10, 10)","(10, 10)","(10, 10)","(10, 11)","(10, 10)","(9, 11)"`

Using the functions in `render.ipynb`, visualizations can be created of the run, actions, and other various aspects. 

Beyond visualizations, `eval_metrics.csv` is created following a run giving the last action, number of steps, when the UGV started moving, number of mines encountered, time, reward, average estimate of traveled path, and total timesteps:

`actions,iterations,ugv_start,mines_encountered,total_time,reward,path_confidence,eval,timesteps`

`[0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.],352,301,15,2812,156.81769580378466,48.3901598003762,True,0`

These can be visualized over time throughout training, or against baseline models in render.ipynb.
