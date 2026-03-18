import json
import os


data_dir = 'data/AgentGym/AgentTraj-L'
traj_files = os.listdir(data_dir)

for traj_file in traj_files:
    with open(os.path.join(data_dir, traj_file), 'r') as f:
        traj = json.load(f)
    print(f"{traj_file}: {len(traj)}")
    