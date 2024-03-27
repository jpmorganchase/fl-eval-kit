# ECIR 2024: Absolute Variation Distance: an Inversion Attack Evaluation Metric for Federated Learning
# Georgios Papadopoulos, Yash Satsangi, Shaltiel Eloul, Marco Pistoia
# JPMorgan Chase

from fl_eval_kit.avd.avd import AttackEvaluationMetrics
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import pickle
from IPython.display import clear_output


PIK = "data/tensors_convergence.dat"
with open(PIK, "rb") as f:
    a=(pickle.load(f))

dm = AttackEvaluationMetrics()

plt.figure(figsize=(12, 8))
index=0

tt = transforms.Compose([transforms.ToTensor()])
tp = transforms.Compose([transforms.ToPILImage()])

for i in range(50, 60):

    for b1 in range(a[0][i].shape[0]):
        v1=tp(torch.abs(a[1][i][b1]))
        v0=tp(a[0][i][b1])

        index+=1
        info_second_batch = []
        mse_batch = []
        for b0 in range(a[0][i].shape[0]):
            mse= (a[0][i][b0]-a[1][i][b1])**2

            info_second_b = dm.AVD(a[0][i][b0], a[1][i][b1])
            info_second_batch.append(float(info_second_b))

            mse_batch.append(float(torch.mean(mse)))

        info_dist_sec = min(info_second_batch)
        info_dist_arg = np.argmin(info_second_batch)

        print("Predicted image")
        plt.imshow(v1)
        plt.show()
        print("Real image")
        plt.imshow(tp(a[0][i][info_dist_arg]))
        plt.show()

        msem=min(mse_batch)

        print("second deriv avd: {0}".format(np.round(info_dist_sec, 5)))
        print("mse: {0}".format(np.round(msem, 5)))

        print()


        flag_measure = bool((float(info_dist_sec))>0.0) # decide if recovered info, emperical measure.
        figure_bin=int(((float(info_dist_sec)))*5.0)+6*np.random.randint(6)+1

        if np.isnan(float(msem)): continue