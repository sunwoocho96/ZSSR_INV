import pytorch
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.gridspec import GridSpec
from configs import Config
from utils import *
from networks import ZSSR_INV, weights_init

class ZSSR:
    kernel = None
    learning_rate = None

    def __init__(self, conf, input, son_input):
        self.conf = conf

        self.ZSSR = ZSSR_INV(conf).cuda()
        self.ZSSR.apply(weights_init)

        self.optimizer = torch.optim.Adam(self.ZSSR.parameters(), lr=conf.lr, betas=(conf.beta1, 0.999))

        self.loss = torch.nn.MSELoss()

        self.input = input
        self.son_input = son_input

        self.mse_rec = []
        self.mse_steps = []
        ##


    def test(self, input):
        output = self.ZSSR.forward(input)

        return output

    def train(self, lr, gt, iteration):

        self.lr = lr.contiguous().cuda()
        self.gt = gt.contiguous().cuda()

        self.optimizer.zero_grad()
        self.hr = self.ZSSR.forward(self.lr)
        loss = self.loss(self.hr, self.gt)
        loss.backward(retain_graph=True)


        self.optimizer.step()

        if iteration % self.conf.run_test_every:
            self.quick_test(iteration)

        self.learning_rate_policy()

        return loss, self.hr


    def quick_test(self, iteration):
        self.rec_input = self.ZSSR.forward(self.son_input)
        self.mse_rec.append(np.mean(np.ndarray.flatten(np.square(self.input - self.rec_input))))

        self.mse_steps.append(iteration)



    def learning_rate_policy(self, iteration):
            # fit linear curve and check slope to determine whether to do nothing, reduce learning rate or finish
            if (not (1 + iteration) % self.conf.learning_rate_policy_check_every
                    and iteration - self.learning_rate_change_iter_nums[-1] > self.conf.min_iters):
                # noinspection PyTupleAssignmentBalance
                [slope, _], [[var, _], _] = np.polyfit(self.mse_steps[-(self.conf.learning_rate_slope_range /
                                                                        self.conf.run_test_every):],
                                                       self.mse_rec[-(self.conf.learning_rate_slope_range /
                                                                      self.conf.run_test_every):],
                                                       1, cov=True)

                # We take the the standard deviation as a measure
                std = np.sqrt(var)

                # Verbose
                print('slope: ', slope, 'STD: ', std)

                # Determine learning rate maintaining or reduction by the ration between slope and noise
                if -self.conf.learning_rate_change_ratio * slope < std:
                    self.learning_rate /= 10
                    print("learning rate updated: ", self.learning_rate)
