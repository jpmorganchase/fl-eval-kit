# ECIR 2024: Absolute Variation Distance: an Inversion Attack Evaluation Metric for Federated Learning
# Georgios Papadopoulos, Yash Satsangi, Shaltiel Eloul, Marco Pistoia
# JPMorgan Chase
import torch
from torchvision import transforms
import numpy as np

class AttackEvaluationMetrics():

    def __init__(self):
        self.tt = transforms.Compose([transforms.ToTensor()])
        self.tp = transforms.Compose([transforms.ToPILImage()])

    def distance_info(self, v0, v1):

        v0=self.tt(self.tp(v0))
        v1=self.tt(self.tp(torch.abs(v1)))

        dv0x=v0[:,:, :-1] - v0[:, :, 1:]
        dv0y=v0[:,:-1, :] - v0[:, 1:, :]

        dv0=torch.abs(dv0x[:, 1:, :]+dv0y[:, :, 1:])/2.0 # How to normalise this value

        dv1x=v1[:,:, :-1] - v1[:, :, 1:]
        dv1y=v1[:,:-1, :] - v1[:, 1:, :]
        dv1=torch.abs(dv1x[:, 1:, :]+dv1y[:, :, 1:])/2.0

        info_dist=torch.mean((dv1-dv0)**2)/0.1 # 0.1 is just to map average noise to 1.0.

        return np.sqrt(info_dist)


    def AVD(self, v0, v1, c1_inv=1.0, c2_inv=0.5):
        """
        v0: real value
        v1: target value
        c1_inv: first order constants
        c2_inv: second order constants
        """

        v0=self.tt(self.tp(v0))
        v1=self.tt(self.tp(torch.abs(v1))) # target value


        dv0x=v0[:,:, :-1] - v0[:, :, 1:]
        dv0y=v0[:,:-1, :] - v0[:, 1:, :]

        # Compute the second derivatives
        d2v0x=dv0x[:,:, :-1] - dv0x[:, :, 1:]
        d2v0y=dv0y[:,:-1, :] - dv0y[:, 1:, :]

        # Normalise the first and second derivatives
        dv0=torch.abs(dv0x[:, 1:, :]+dv0y[:, :, 1:])
        d2v0=torch.abs(d2v0x[:, 2:, :]+d2v0y[:, :, 2:])

        dv1x=v1[:,:, :-1] - v1[:, :, 1:]
        dv1y=v1[:,:-1, :] - v1[:, 1:, :]

        d2v1x=dv1x[:,:, :-1] - dv1x[:, :, 1:]
        d2v1y=dv1y[:,:-1, :] - dv1y[:, 1:, :]

        dv1=torch.abs(dv1x[:, 1:, :]+dv1y[:, :, 1:])
        d2v1=torch.abs(d2v1x[:, 2:, :]+d2v1y[:, :, 2:])

        info_dist=c1_inv*torch.mean((dv1-dv0)**2) + c2_inv*torch.mean((d2v1-d2v0)**2)

        return np.sqrt(info_dist)


