import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.math.fw_attack_step import AttackStep
from src.math.lmo import LMO
from src.params import RANDOM_SEED


def run_attack(target_model, test_loader, epsilons, iterations_number, alg='fw', norm_p="l_inf"):
    model = target_model.model
    device = target_model.device

    history = []
    accuracies = []
    times = []

    print(f"Attack Method: {alg}\tNorm: {norm_p}")

    for epsilon in tqdm(epsilons):
        correct = 0
        attacks_number = 0
        start = time.process_time()

        for x_0, target in test_loader:
            x_t = x_0.detach().clone().to(device)
            x_0, target = x_0.to(device), target.to(device)

            criterion = nn.NLLLoss()
            lmo = LMO(epsilon, x_0, norm_p)
            attack_step = AttackStep(alg, lmo)
            convergence = None
            stats = {}

            output = model(x_t)
            init_pred = output.max(1, keepdim=True)[1]
            # If the network is already wrong, skip this example
            if init_pred.item() != target.item():
                continue

            for t in range(iterations_number):
                iter_start_time = time.process_time()
                x_t.requires_grad = True

                output = model(x_t)
                loss = -criterion(output, target)  # negative log used
                model.zero_grad()
                loss.backward()

                if alg == 'fw_momentum_blackbox':
                    # Hard-coded values for now
                    b = 20
                    delta = 0.001
                    x_t_grad = estimate_grad(model, x_t, target, b, delta)
                else:
                    x_t_grad = x_t.grad

                # Call Attack
                with torch.no_grad():
                    perturbed_image, convergence = attack_step.step(x_t, x_t_grad, t)

                x_t = perturbed_image

                # Re-classify the perturbed image
                x_t.requires_grad = False
                output = model(x_t)
                stats['l_inf'] = torch.max(torch.abs(x_0 - perturbed_image)).item()
                stats['model_loss'] = loss.item()

                # Check for success
                final_pred = output.max(1, keepdim=True)[1]
                if final_pred.item() != target.item():
                    success = True
                else:
                    success = False
                    correct += 1
                attacks_number += 1

                # metric logging
                iter_end_time = time.process_time()
                iter_time_taken = iter_end_time - iter_start_time
                history_iter = {
                    'example_idx': attacks_number,
                    'iter': t + 1,
                    'convergence': convergence if convergence is not None else None,
                    'success': success,
                    'target': target.item(),
                    'pred': final_pred.item(),
                    'iter_time': iter_time_taken,
                    'epsilon': epsilon,
                }

                if stats is not None:
                    history_iter.update(stats)
                history.append(history_iter)

            end = time.process_time()
            time_taken = end - start
            times.append(time_taken)

        # Calculate final accuracy for this epsilon
        final_accuracy = correct / attacks_number
        accuracies.append(final_accuracy)

        average_time = round(sum(times) / len(times), 4)

        # print(f"Attack Method: {alg}\tNorm: {norm_p}")
        # print(
        #     f"Epsilon: {epsilon}\tCorrect Classifications (Failed Attacks) = {correct} / {attacks_number} "
        #     f"= {final_accuracy}" + f"\tLast epsilon time: {time_taken:.2f}s"
        # )

    return accuracies, average_time, pd.DataFrame(history)


def estimate_grad(model, x, target, b, delta, option='II'):
    torch.manual_seed(RANDOM_SEED)
    q = torch.zeros_like(x)

    def f(x):
        with torch.no_grad():
            output = model(x)
            loss = -F.nll_loss(output, target)
        return loss.item()

    for _ in range(b):
        # if option == 'I':
        #     # Опция I: Сэмплирование u_i с единичной L2-нормой
        #     u_i = torch.randn_like(x)
        #     u_i = u_i / u_i.norm(p=2)

        #     f_plus = f(x + delta * u_i)
        #     f_minus = f(x - delta * u_i)

        #     fi = (f_plus - f_minus) * u_i * d / (2 * delta * b)
        #     q += fi
        if option == 'II':
            # Опция II: Сэмплирование u_i из стандартного нормального распределения
            u_i = torch.randn_like(x)

            f_plus = f(x + delta * u_i)
            f_minus = f(x - delta * u_i)

            fi = (f_plus - f_minus) * u_i / (2 * delta * b)
            q += fi
        else:
            raise ValueError("Неверная опция для сэмплирования u_i. Выберите 'I' или 'II'.")

    return q
