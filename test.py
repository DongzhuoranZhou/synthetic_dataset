import torch
import numpy as np
# G_old = torch.load("dataset/syn2/width_1/G_1000_pairs_depth_16_width_1_hdim_16_high_gap_True_backup.pt")
# G_new = torch.load("dataset/syn2/width_1/G_1000_pairs_depth_16_width_1_hdim_16_high_gap_True.pt")
A = np.array([[0,1,0,0,0,0,0,0],[1,0,1,0,0,0,0,0],[0,1,0,1,0,0,0,1],[0,0,1,0,1,1,1,0],[0,0,0,1,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,1,0,0,0,0,0]])
A_2 = np.matmul(A,A)
A_3 = np.matmul(A_2,A)
A_3_inverse = np.matmul(A,A_2)
pass