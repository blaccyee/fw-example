import torch


class AttackStep:
    def __init__(self, method, lmo, momentum=0.8):
        self.method = method
        self.lmo = lmo
        self.momentum = momentum
        self.m_t_last = None

    def step(self, x_t, x_t_grad, k=1):
        if self.method == 'fw':
            return self.fw_step(x_t, x_t_grad, k)
        elif self.method == 'fw_momentum' or self.method == 'fw_momentum_blackbox':
            return self.fw_step_momentum(x_t, x_t_grad, self.momentum, k)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def fw_step(self, x_t, g_t, k):
        v_t = self.lmo.get(g_t)
        d_t = v_t - x_t
        self.d_t = d_t

        stepsize = 2 / (k + 2)

        perturbed_image = x_t + stepsize * d_t
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        convergence = torch.sum(-d_t * g_t).item()
        return perturbed_image, convergence

    def fw_step_momentum(self, x_t, g_t, momentum=0.8, k=1):
        m_t = (1 - momentum) * g_t
        if self.m_t_last is not None:
            m_t += momentum * self.m_t_last
        v_t = self.lmo.get(m_t)
        d_t = v_t - x_t

        stepsize = 2 / (k + 2)

        convergence = torch.sum(-d_t * g_t).item()
        perturbed_image = x_t + stepsize * d_t
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        self.m_t_last = m_t.clone().detach()
        return perturbed_image, convergence
