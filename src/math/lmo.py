import torch


class LMO:
    def __init__(self, epsilon, x_0, p_norm):
        self.x_0 = x_0.clone().detach()
        self.epsilon = epsilon
        self.p_norm = p_norm
        self.methods = {
            "l_inf": self._lmo_inf,
            "l_1": self._lmo_l1,
            "l_2": self._lmo_l2,
        }
        self.method = self.methods.get(p_norm, None)
        if self.method is None:
            raise Exception(f"Unsupported norm: {p_norm}. Try one of these: {list(self.methods.keys())}")

    def get(self, gradient):
        return self.method(gradient)

    def _lmo_inf(self, gradient):
        return -self.epsilon * gradient.sign() + self.x_0

    def _lmo_l1(self, gradient):
        abs_gradient = gradient.abs()
        sign_gradient = gradient.sign()
        perturbation = torch.zeros_like(gradient)

        for i in range(gradient.size(0)):
            # Find the index of the maximum absolute value in each row
            _, idx = torch.topk(abs_gradient[i].view(-1), 1)
            # Set the corresponding element in the perturbation tensor
            perturbation[i].view(-1)[idx] = sign_gradient[i].view(-1)[idx]

        return self.epsilon * perturbation

    def _lmo_l2(self, gradient):
        return -self.epsilon * gradient / torch.norm(gradient, p=2, dim=-1, keepdim=True) + self.x_0
