import argparse
import torch
import os
import scipy.io as sio
import numpy as np


class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.conf = None

