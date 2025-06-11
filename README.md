# Breaking Data Silos:Towards Open and Scalable Mobility Foundation Models via Generative Continual Learning
<!-- add image -->
<p align="center">
  <img src="fig/MoveGCL.svg" alt="WorldMove Logo"/>
</p>
MoveGCL is a scalable and privacy-preserving framework for training mobility foundation models via generative continual learning. Without sharing raw data, MoveGCL enables decentralized and progressive model evolution by replaying synthetic trajectories generated from a frozen teacher model, and reinforces knowledge retention through a tailored distillation strategy that mitigates catastrophic forgetting. To address the heterogeneity of mobility patterns, MoveGCL incorporates a Mixture-of-Experts Transformer with a mobility-aware expert routing mechanism, and employs a layer-wise progressive adaptation strategy to stabilize continual updates.

## Data set
Our trajectory data is stored in the <code>traj_data</code> directory. Each line in the text files represents the trajectory of a single user over three consecutive days, formatted as follows: 
<code>1391097 0 8 1104,0,0,0,0;1137,0,9,9,1;1137,1,0,39,0;1137,2,3,51,0;1103,2,17,14,1;1137,2,22,5,1</code> 
In this example: 
- <code>1391097</code> is the user ID.
- <code>0</code> is the quantized radius of gyration, denoted as <code>r_gyr</code>.
- <code>8</code> is the quantized location entropy, denoted as <code>H_loc</code>.
- Each following entry, such as <code>1104,0,0,0,0</code>, represents a single point in the trajectory, with the fields defined as:
  | Field         | Description                                |
  | ------------- | ------------------------------------------ |
  | `location_id` | Unique location identifier                 |
  | `day_of_week` | 0 = Monday … 6 = Sunday                    |
  | `time_slot`   | Index of the time interval within the day  |
  | `t_wait`      | Waiting time at that location              |
  | `d_jump`      | Distance jumped from the previous location |

