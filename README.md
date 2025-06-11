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

## ⚙️ Installation
### Environment
### Dependencies

## 🏃 Model Training
在训练前注意先调整文件中的路径
### Stage-1 train base model
训练base model的代码在'/train_base_model.py'中，你可以通过 <code>python ./MoveGCL/train_base_model.py --n_embd 512 --n_layer 6 --num_experts 4 --B 16 --city 'WashingtonDC' 'Seattle' 'Atlanta' --train_root '/data0/liuyukun/MoveGCL/traj_data/train' --val_root '/data0/liuyukun/MoveGCL/traj_data/val' --test_root '/data0/liuyukun/MoveGCL/traj_data/test' --epoch 30 --lr 1.2e-5</code> 运行这段代码，其中<code>n_embd</code>表示MoE transformer的隐藏维度，<code>n_layer</code>表示MoE transformer的层数，<code>num_experts</code>表示每一层MoE transformer中experts的数量。运行完后训练得到的模型会被存储到'./base_model'中
### Stage-2 Generative continual learning
这部分的代码在'./GCL_data'中
### Generate pseudo-trajectories
- 获取每座城市的经验分布，每当模型学习一座新城市时，你需要在这座城市的数据上运行'./GCL_data/get_first_loc_distribute.py',它会提取这座城市的轨迹中的empirical distribution of first locations conditioned on different length（对应文论文中的公式4），并存储在'./MoveGCL/GCL_data/data_distribution'中。
- 为生成伪轨迹做准备，你首先需要在当前城市的数据中进行采样（论文中公式3对应的轨迹），运行'./GCL_data/get_sample_data.py'，它会将采样的结构存储在'./GCL_data/sampled_data'中，然后你需要运行'./GCL_data/replace_first_loc.py'，它会采在之前收集的检验分布中采样一些，并将'./GCL_data/sampled_data'中存储的轨迹的第一个点替换掉，替换的结果存储在'./MoveGCL/GCL_data/replaced_first_loc_data'。
- 生成伪轨迹（对应论文中的公式5），运行"./MoveGCL/GCL_data/gen_pseudo_traj.py"，它会将生成的伪轨迹存储在'./MoveGCL/GCL_data/pseudo_traj'中
### Get expert to froze
Generative continual learning的Layer-Wise Progressive Adaptation中需要获模型中的哪些Experts需要被冻结，运行'./MoveGCL/get_experts_to_forze.py'，然后你需要在模型文件夹下找到文件'{model_file}/froze_info_file/layer_max_indices.txt'，这个文件中从上到下分别记录了最靠近输入到最靠近输出的层需要冻结的专家
### Continual learning
持续学习的代码在'./MoveGCL/continual_learning.py'中
