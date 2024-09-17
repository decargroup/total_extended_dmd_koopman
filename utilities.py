import numpy as np
import pykoop.lmi_regressors
import scipy
from numpy import random
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict
from pykoop.lmi_regressors import LmiRegressor
from pykoop import lmi_regressors
from typing import Any, Dict, Optional
from pykoop import tsvd
import logging
import tempfile
import joblib
import picos
import scipy.linalg
import scipy.signal
import sklearn.base
import os
from random import randint
import pykoop
import shutil
import pickle
import statistics
import pykoop
import sklearn
from sklearn.metrics import mean_squared_error

polite_stop = False

# Create logger
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

# Create temporary cache directory for memoized computations
_cachedir = tempfile.TemporaryDirectory(prefix='pykoop_')
log.info(f'Temporary directory created at `{_cachedir.name}`')
memory = joblib.Memory(_cachedir.name, verbose=0)

color_dict = {
    'EDMD': (0.90, 0.60, 0.00),
    'EDMD-AS': (0.80, 0.40, 0.00),
    'FBEDMD': (0.35, 0.70, 0.90),
    'FBEDMD-AS': (0.00, 0.45, 0.70),
    'TEDMD': (0.95, 0.90, 0.25),
    'TEDMD-AS': (0.00, 0.60, 0.50)
}
color_dict2 = {
    'EDMD': (0.90, 0.60, 0.00),
    'Af': (0.80, 0.40, 0.00),
    'Ab': (0.35, 0.70, 0.90),
    'A': (0.00, 0.45, 0.70)
}
color_list = [(0.00, 0.45, 0.70), (0.90, 0.60, 0.00), (0.00, 0.60, 0.50),
              (0.35, 0.70, 0.90), (0.60, 0.60, 0.60), (0.95, 0.90, 0.25)]
linestyle_dict = {
    'EDMD': '-',
    'EDMD-AS': '-',
    'FBEDMD': '-',
    'FBEDMD-AS': '-',
    'TEDMD': '-',
    'TEDMD-AS': '-'
}


def Q_gen(n_states: int = 2,
          add_noise: list = [0, 1],
          noise: int = 0) -> np.ndarray:

    Q = np.zeros((n_states, n_states))

    for i in add_noise:
        Q[i, i] = noise

    return Q


def add_noise(data, Q, mu, n, seed: int = 3) -> np.ndarray:

    v = np.zeros((Q.shape[0], data.shape[1]))

    e, u = np.linalg.eig(Q)

    rng = random.default_rng(seed=seed)

    r_noise = rng.normal(0, 1, (data.shape[0], data.shape[1])) + mu

    v = np.diag(e)**0.5 @ r_noise

    noise = u @ v

    Data = noise + data

    n = data.shape[1]

    s_mean = np.mean(data)
    n_mean = np.mean(noise)
    s_pow = np.sum((data - s_mean)**2) / n
    n_pow = np.sum((noise - n_mean)**2) / n
    snr = 10 * np.log10(s_pow / n_pow)

    print('Avg signal power: {}, avg noise power: {}, SNR: {}'.format(
        s_pow, n_pow, snr))

    return Data.T, snr, noise


import numpy as np
from scipy.integrate import quad


def optimal_SVHT_coef(beta, sigma_known):
    """
    Coefficient determining optimal location of Hard Threshold for Matrix
    Denoising by Singular Values Hard Thresholding when noise level is known or
    unknown.

    Args:
        beta (float or numpy.ndarray): Aspect ratio m/n of the matrix to be denoised, 0 < beta <= 1.
        sigma_known (bool): 1 if noise level known, 0 if unknown.

    Returns:
        numpy.ndarray: Optimal location of hard threshold, up the median data singular
        value (sigma unknown) or up to sigma*sqrt(n) (sigma known);
        a numpy array of the same dimension as beta, where coef[i] is the 
        coefficient corresponding to beta[i].
    """
    if sigma_known:
        coef = optimal_SVHT_coef_sigma_known(beta)
    else:
        coef = optimal_SVHT_coef_sigma_unknown(beta)

    return coef


def optimal_SVHT_coef_sigma_known(beta):
    """
    Computes the optimal SVHT coefficient when the noise level is known.

    Args:
        beta (float or numpy.ndarray): Aspect ratio m/n of the matrix to be denoised, 0 < beta <= 1.

    Returns:
        numpy.ndarray: Optimal SVHT coefficient for the given beta.
    """
    beta = np.asarray(beta)
    assert np.all(beta > 0)
    assert np.all(beta <= 1)

    w = (8 * beta) / (beta + 1 + np.sqrt(beta**2 + 14 * beta + 1))
    lambda_star = np.sqrt(2 * (beta + 1) + w)
    return lambda_star


def optimal_SVHT_coef_sigma_unknown(beta):
    """
    Computes the optimal SVHT coefficient when the noise level is unknown.

    Args:
        beta (float or numpy.ndarray): Aspect ratio m/n of the matrix to be denoised, 0 < beta <= 1.

    Returns:
        numpy.ndarray: Optimal SVHT coefficient for the given beta when noise level is unknown.
    """
    beta = np.asarray(beta)
    assert np.all(beta > 0)
    assert np.all(beta <= 1)

    coef = optimal_SVHT_coef_sigma_known(beta)

    MPmedian = np.zeros_like(beta)
    for i in range(len(beta)):
        MPmedian[i] = MedianMarcenkoPastur(beta[i])

    omega = coef / np.sqrt(MPmedian)
    return omega


def MarcenkoPasturIntegral(x, beta):
    """
    Computes the Marcenko-Pastur integral for a given x and beta.

    Args:
        x (float): Value within the range of integration.
        beta (float): Aspect ratio m/n of the matrix to be denoised, 0 < beta <= 1.

    Returns:
        float: Value of the Marcenko-Pastur integral.
    """
    assert 0 < beta <= 1

    lobnd = (1 - np.sqrt(beta))**2
    hibnd = (1 + np.sqrt(beta))**2

    assert lobnd <= x <= hibnd

    def dens(t):
        return np.sqrt((hibnd - t) * (t - lobnd)) / (2 * np.pi * beta * t)

    I, _ = quad(dens, lobnd, x)
    return I


def MedianMarcenkoPastur(beta):
    """
    Computes the median of the Marcenko-Pastur distribution for a given beta.

    Args:
        beta (float): Aspect ratio m/n of the matrix to be denoised, 0 < beta <= 1.

    Returns:
        float: Median of the Marcenko-Pastur distribution.
    """
    MarPas = lambda x: 1 - incMarPas(x, beta, 0)
    lobnd = (1 - np.sqrt(beta))**2
    hibnd = (1 + np.sqrt(beta))**2
    change = True
    while change and (hibnd - lobnd > 0.001):
        change = False
        x = np.linspace(lobnd, hibnd, 5)
        y = np.array([MarPas(xi) for xi in x])
        if np.any(y < 0.5):
            lobnd = np.max(x[y < 0.5])
            change = True
        if np.any(y > 0.5):
            hibnd = np.min(x[y > 0.5])
            change = True
    med = (hibnd + lobnd) / 2
    return med


def incMarPas(x0, beta, gamma):
    """
    Computes the incomplete Marcenko-Pastur integral.

    Args:
        x0 (float): Upper limit of integration.
        beta (float): Aspect ratio m/n of the matrix to be denoised, 0 < beta <= 1.
        gamma (float): Exponential factor.

    Returns:
        float: Value of the incomplete Marcenko-Pastur integral.
    """
    assert beta <= 1

    topSpec = (1 + np.sqrt(beta))**2
    botSpec = (1 - np.sqrt(beta))**2

    def MarPas(x):
        return np.sqrt((topSpec - x) * (x - botSpec)) / (beta * x) / (
            2 * np.pi) if (topSpec - x) * (x - botSpec) > 0 else 0

    if gamma != 0:
        fun = lambda x: x**gamma * MarPas(x)
    else:
        fun = MarPas

    I, _ = quad(fun, x0, topSpec)
    return I


def sv_finder(variance: float, robot: str):

    # In the TLSDMDC paper, they recommend to take p_theta + n _inputs singular values, but it is too many. In the SVHT paper, they also recommend too many.
    # Here we find the amount of singular values required to drop the the condition number of Psi @ Q.

    # For nl_msd with poly2_centers10
    if robot == 'nl_msd':
        if variance == 0.00001:
            sv_to_keep3 = 7
        elif variance == 0.0001:
            sv_to_keep3 = 7
        elif variance == 0.001:
            sv_to_keep3 = 7
        elif variance == 0.01:
            sv_to_keep3 = 7
        elif variance == 0.1:
            sv_to_keep3 = 7
        elif variance == 1:
            sv_to_keep3 = 4
        elif variance == 0:
            sv_to_keep3 = 7

    # For soft_robot with poly2_centers10
    if robot == 'soft_robot':
        if variance == 0.00001:
            sv_to_keep3 = 13
        elif variance == 0.0001:
            sv_to_keep3 = 13
        elif variance == 0.001:
            sv_to_keep3 = 13
        elif variance == 0.01:
            sv_to_keep3 = 12
        elif variance == 0.1:
            sv_to_keep3 = 8
        elif variance == 1:
            sv_to_keep3 = 6
        elif variance == 0:
            sv_to_keep3 = 13

    return sv_to_keep3


def plot_rms_and_avg_error_paper(
    val_data: Dict[str, np.ndarray],
    true_val_data: np.ndarray,
    path: str,
    norm_params: np.ndarray,
    robot: str,
    val: int = 0,
    n: int = None,
    n_bins: int = 20,
    variance: float = 0.01,
    **kwargs,
) -> None:

    plt.rcParams.update(**kwargs)
    usetex = True if shutil.which('latex') else False
    if usetex:
        plt.rc('text', usetex=True)

    fig, ax = plt.subplots(
        1,
        2,
        constrained_layout=True,
        figsize=(6.5, 2.5),
        sharey=True,
    )
    i = 0

    conv_param = 2.54 if robot == 'soft_robot' else 1

    if robot == 'soft_robot':
        if n is None:
            n = val_data[list(val_data.keys())[0]].shape[0]

        n_states = val_data[list(val_data.keys())[0]].shape[1] - 1
        n_val_eps = int(np.max(true_val_data[:, 0]) + 1)

        error_metrics = [r'RMSE', r'MAE']
        for metric in error_metrics:
            avg_error = {}
            error = {}
            ep_val = {}
            for tag in val_data.keys():
                if tag == 'EDMD' or tag == 'FBEDMD' or tag == 'TEDMD':
                    continue
                avg_error[tag] = 0
            for val in range(n_val_eps):
                ep_val['{}'.format(val)] = {}
            max_error = 0
            # regressors = [r'EDMD-AS', r'FBEDMD-AS']
            regressors = [r'EDMD-AS', r'FBEDMD-AS', r'TEDMD-AS']
            for ep in range(n_val_eps):

                for tag, data in val_data.items():
                    if tag == r'EDMD' or tag == r'FBEDMD' or tag == r'TEDMD':
                        continue
                    if metric == r'RMSE':
                        temp = np.sqrt(
                            np.sum(
                                np.linalg.norm(
                                    data[data[:, 0] == ep, 1:][:n] *
                                    (conv_param * norm_params[0]) -
                                    true_val_data[true_val_data[:, 0] == ep, 1:
                                                  (n_states + 1)][:n] *
                                    (conv_param * norm_params[0]),
                                    2,
                                    axis=1)**2) / n)
                        test = data[data[:, 0] == ep, 1:][:n] * (
                            conv_param * norm_params[0]
                        ) - true_val_data[true_val_data[:, 0] == ep, 1:(
                            n_states + 1)][:n] * (conv_param * norm_params[0])
                    elif metric == r'MAE':
                        temp = np.sum(
                            np.linalg.norm(
                                data[data[:, 0] == ep, 1:][:n] *
                                (conv_param * norm_params[0]) -
                                true_val_data[true_val_data[:, 0] == ep, 1:
                                              (n_states + 1)][:n] *
                                (conv_param * norm_params[0]),
                                1,
                                axis=1)) / n
                    max_error = temp if temp > max_error else max_error
                    error[tag] = temp
                    avg_error[tag] += temp / n_val_eps
                ep_val['{}'.format(ep)] = list(error.values())

            for tag in val_data.keys():
                if tag == r'EDMD' or tag == r'FBEDMD' or tag == r'TEDMD':
                    continue
                print('Average {} for {}: {}'.format(metric, tag,
                                                     avg_error[tag]))

            x_ticks = np.arange(0, len(error.keys()), 1)
            y_ticks = np.arange(0, max_error, max_error / 10)
            width = 0.1
            multiplier = 0

            for ep_tag, data in ep_val.items():
                offset = width * multiplier
                rects = ax[i].bar(x_ticks + offset,
                                  np.abs(data),
                                  width,
                                  edgecolor='k',
                                  linewidth=0.5,
                                  color=color_list[int(ep_tag)],
                                  label=r'test ep. {}'.format(int(ep_tag) + 1),
                                  zorder=3)
                multiplier += 1

            offset = width * multiplier
            rects = ax[i].bar(x_ticks + offset,
                              np.abs(list(avg_error.values())),
                              width,
                              edgecolor='k',
                              linewidth=0.5,
                              hatch='//',
                              color=color_list[int(ep_tag) + 1],
                              label='average',
                              zorder=3)

            if robot == 'soft_robot':
                ax[i].set_ylabel('{}'.format(metric) + r' (cm)')
            else:
                ax[i].set_ylabel('{}'.format(metric) + r' (mm)')
            if metric == r'RMSE':
                if variance == 0.01:
                    ax[i].set_ylim(0, 1.2)
                    ax[i].set_yticks(np.linspace(0, 1.4, 8))
                if variance == 0.1:
                    ax[i].set_ylim(0, 4)
                    ax[i].set_yticks(np.linspace(0, 5.5, 12))

            ax[i].grid(axis='y', linestyle='--', zorder=0)
            i = +1

        ax[1].set_xticks(x_ticks + width * 2.0, regressors)
        ax[0].set_xticks(x_ticks + width * 2.0, regressors)

        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            bbox_to_anchor=(0.5, 0.03),
            loc='upper center',
            ncol=5,
        )

        fig.text(0.034, 1.015, r'$(a)$')
        fig.text(0.54, 1.015, r'$(b)$')

        fig.savefig('build/figures/paper/{}_error_bars_{}.pdf'.format(
            robot, variance),
                    bbox_inches='tight')
        fig.savefig('build/figures/paper/{}_error_bars_{}.png'.format(
            robot, variance),
                    bbox_inches='tight')
        fig.tight_layout()


def plot_trajectory_error_paper(
    val_data: Dict[str, np.ndarray],
    true_val_data: np.ndarray,
    path: str,
    norm_params: np.ndarray,
    robot: str,
    val: int = 2,
    n: int = None,
    variance: float = 0.01,
    **kwargs,
) -> None:

    if n is None:
        n = val_data[list(val_data.keys())[0]].shape[0]

    n_val_eps = int(np.max(true_val_data[:, 0]) + 1)

    plt.rcParams.update(**kwargs)
    usetex = True if shutil.which('latex') else False
    if usetex:
        plt.rc('text', usetex=True)

    fig_dict = {}

    conv_param = 2.54 if robot == 'soft_robot' else 1

    for i in range(n_val_eps):
        fig, axs = plt.subplot_mosaic(
            [['x1_err', 'traj'], ['x2_err', 'traj']],
            figsize=(5.3348, 2.2),
            layout='constrained',
        )
        max_y = 0
        min_y = 0
        max_x = 0
        min_x = 0

        ax = axs['traj']

        # Plot trajectory
        for tag, data in val_data.items():

            if tag == 'EDMD' or tag == 'FBEDMD' or tag == 'TEDMD':
                continue

            if np.isnan(data[data[:, 0] == i, 1][:n]).any() or np.isnan(
                    data[data[:, 0] == i, 2][:n]).any():
                print('NaN detected in {}.'.format(tag))
                continue
            else:
                ax.plot(data[data[:, 0] == i, 1][:n] *
                        (conv_param * norm_params[0]),
                        data[data[:, 0] == i, 2][:n] *
                        (conv_param * norm_params[1]),
                        label=tag,
                        color=color_dict[tag],
                        linestyle=linestyle_dict[tag])
            temp_x = data[data[:, 0] == i,
                          1][:n] * (conv_param * norm_params[0])
            temp_y = data[data[:, 0] == i,
                          2][:n] * (conv_param * norm_params[1])
            max_x = np.max(temp_x) if (np.max(temp_x) > max_x) else max_x
            min_x = np.min(temp_x) if (np.min(temp_x) < min_x
                                       and np.min(temp_x) < 0) else min_x
            max_y = np.max(temp_y) if (np.max(temp_y) > max_y) else max_y
            min_y = np.min(temp_y) if (np.min(temp_y) < min_y
                                       and np.min(temp_y) < 0) else min_y

        ax.plot(true_val_data[true_val_data[:, 0] == i, 1][:n] *
                (conv_param * norm_params[0]),
                true_val_data[true_val_data[:, 0] == i, 2][:n] *
                (conv_param * norm_params[1]),
                label='Ground truth',
                color='k',
                linestyle=':',
                zorder=5)

        if i == val:

            ax.scatter(true_val_data[true_val_data[:, 0] == i, 1][0] *
                       (conv_param * norm_params[0]),
                       true_val_data[true_val_data[:, 0] == i, 2][0] *
                       (conv_param * norm_params[1]),
                       marker='x',
                       s=25,
                       color='k',
                       zorder=4)

            ax.scatter(true_val_data[true_val_data[:, 0] == i, 1][n - 1] *
                       (conv_param * norm_params[0]),
                       true_val_data[true_val_data[:, 0] == i, 2][n - 1] *
                       (conv_param * norm_params[1]),
                       marker='x',
                       s=25,
                       color='k',
                       zorder=4)

        if robot == 'soft_robot':
            ax.set(ylabel=r'$x_{}$ (cm)'.format(2))
            ax.set(xlabel=r'$x_{}$ (cm)'.format(1))
        else:
            ax.set(ylabel=r'$x_{}$ (mm)'.format(2))
            ax.set(xlabel=r'$x_{}$ (mm)'.format(1))

        if robot == 'soft_robot':
            if val == 3:
                ax.set_ylim(0, 20)
                ax.set_xlim(0, 20)
                ax.set_xticks(np.linspace(0, 20, 11))
                ax.set_yticks(np.linspace(0, 20, 11))
            if val == 2:
                ax.set_ylim(0, 20)
                ax.set_xlim(0, 20)
                ax.set_xticks(np.linspace(0, 20, 11))
                ax.set_yticks(np.linspace(0, 20, 11))
            if val == 1:
                ax.set_ylim(-12, -2)
                ax.set_yticks(np.linspace(-12, -2, 6))
                if variance == 0.01:
                    ax.set_xlim(-2, 10)
                    ax.set_xticks(np.linspace(-2, 10, 7))
                if variance == 0.1:
                    ax.set_xlim(-4, 10)
                    ax.set_xticks(np.linspace(-4, 10, 8))
            if val == 0:
                ax.set_xlim(-16, -6)
                ax.set_ylim(-6, 6)
                ax.set_xticks(np.linspace(-16, -6, 6))
                ax.set_yticks(np.linspace(-6, 6, 7))
        elif robot == 'nl_msd':
            if val == 0:
                ax.set_ylim(-4, 6)
                ax.set_xlim(-4, 6)
                ax.set_xticks(np.linspace(-4, 6, 6))
                ax.set_yticks(np.linspace(-4, 6, 6))
            if val == 1:
                ax.set_ylim(-6, 6)
                ax.set_xlim(-6, 6)
                ax.set_xticks(np.linspace(-6, 6, 7))
                ax.set_yticks(np.linspace(-6, 6, 7))

        ax.set_aspect('equal')

        ax.grid(linestyle='--')

        # Plot prediction errors
        ax = [axs['x1_err'], axs['x2_err']]

        t = np.linspace(0, n * 0.01, n) if robot == 'nl_msd' else np.linspace(
            0, n * 0.0829, n)

        conv_param = 2.54 if robot == 'soft_robot' else 1
        for tag, data in val_data.items():

            if tag == 'EDMD' or tag == 'FBEDMD' or tag == 'TEDMD':
                continue

            for k in range(2):

                true_temp = true_val_data[true_val_data[:, 0] == i,
                                          k + 1][:n] * (conv_param *
                                                        norm_params[k])
                pred_temp = data[data[:, 0] == i,
                                 k + 1][:n] * (conv_param * norm_params[k])

                if np.isnan(pred_temp).any():
                    print('NaN detected in {} state {}'.format(tag, k))
                    continue
                else:

                    if i == val:
                        ax[k].plot(
                            t,
                            pred_temp - true_temp,
                            color=color_dict[tag],
                            label=tag,
                            linestyle=linestyle_dict[tag],
                        )
                    else:
                        ax[k].plot(
                            pred_temp - true_temp,
                            color=color_dict[tag],
                            label=tag,
                        )

                if robot == 'soft_robot':
                    ax[k].set(ylabel='$\Delta x_{}$ (cm)'.format(k + 1))
                else:
                    ax[k].set(ylabel='$\Delta x_{}$ (mm)'.format(k + 1))

        if robot == 'nl_msd':
            if val == 0:
                ax[1].set_xlim(0, n * 0.01)
                ax[1].set_xticks(np.linspace(0, 20, 11))
                if variance == 0.01:
                    ax[0].set_ylim(-3, 3)
                    ax[0].set_yticks(np.linspace(-3, 3, 7))
                    ax[1].set_ylim(-3, 3)
                    ax[1].set_yticks(np.linspace(-3, 3, 7))
                if variance == 0.1:
                    ax[0].set_ylim(-5, 3)
                    ax[0].set_yticks(np.linspace(-5, 3, 5))
                    ax[1].set_ylim(-3, 5)
                    ax[1].set_yticks(np.linspace(-3, 5, 5))
                fig.text(0.71, 0.85, r'$t_\mathrm{i}$', fontsize=14)
                fig.text(0.76, 0.28, r'$t_\mathrm{f}$', fontsize=14)
            if val == 1:
                ax[0].set_ylim(-5, 5)
                ax[0].set_yticks(np.linspace(-4, 4, 5))
                ax[1].set_ylim(-5, 5)
                ax[1].set_yticks(np.linspace(-4, 4, 5))
                ax[1].set_xlim(0, n * 0.01)
                ax[1].set_xticks(np.linspace(0, 20, 11))
                fig.text(0.65, 0.88, r'$t_\mathrm{i}$', fontsize=14)
                fig.text(0.75, 0.36, r'$t_\mathrm{f}$', fontsize=14)
        elif robot == 'soft_robot':
            if val == 3:
                ax[0].set_ylim(-2, 0.7)
                ax[0].set_yticks(np.arange(-2, 1.1, 0.5))
                ax[1].set_ylim(-0.4, 1.1)
                ax[1].set_yticks(np.arange(-0.6, 1.3, 0.3))
                ax[1].set_xlim(0, n * 0.0829)
                ax[1].set_xticks(np.arange(0, 26, 5))
                fig.text(0.70, 0.38, r'$t_\mathrm{i}$', fontsize=14)
                fig.text(0.68, 0.59, r'$t_\mathrm{f}$', fontsize=14)
            if val == 2:
                ax[0].set_ylim(-3, 3)
                ax[0].set_yticks(np.linspace(-3, 3, 7))
                ax[1].set_ylim(-3, 3)
                ax[1].set_yticks(np.linspace(-3, 3, 7))
                ax[1].set_xlim(0, n * 0.0829)
                ax[1].set_xticks(np.arange(0, 26, 5))
                fig.text(0.64, 0.54, r'$t_\mathrm{i}$', fontsize=14)
                fig.text(0.74, 0.51, r'$t_\mathrm{f}$', fontsize=14)
            if val == 1:
                ax[1].set_xlim(0, n * 0.0829)
                ax[1].set_xticks(np.arange(0, 26, 5))
                if variance == 0.01:
                    ax[0].set_ylim(-0.3, 0.7)
                    ax[0].set_yticks(np.arange(-0.3, 0.8, 0.2))
                    ax[1].set_ylim(-0.2, 1)
                    ax[1].set_yticks(np.arange(-0.2, 1.1, 0.2))
                    fig.text(0.73, 0.27, r'$t_\mathrm{i}$', fontsize=14)
                    fig.text(0.93, 0.52, r'$t_\mathrm{f}$', fontsize=14)
                if variance == 0.1:
                    ax[0].set_ylim(-2.6, 1.2)
                    ax[0].set_yticks(np.arange(-2.6, 1.2, 0.6))
                    ax[1].set_ylim(-0.4, 4.4)
                    ax[1].set_yticks(np.arange(-0.4, 4.4, 0.8))
                    fig.text(0.75, 0.33, r'$t_\mathrm{i}$', fontsize=14)
                    fig.text(0.93, 0.52, r'$t_\mathrm{f}$', fontsize=14)
            if val == 0:
                ax[0].set_ylim(-2, 0.4)
                ax[0].set_yticks(np.arange(-2, 0.5, 0.4))
                ax[1].set_ylim(-1.2, 0.7)
                ax[1].set_yticks(np.arange(-1.2, 0.7, 0.3))
                ax[1].set_xlim(0, n * 0.0829)
                ax[1].set_xticks(np.arange(0, 12, 2))
                fig.text(0.67, 0.48, r'$t_\mathrm{i}$', fontsize=14)
                fig.text(0.83, 0.85, r'$t_\mathrm{f}$', fontsize=14)

        ax[1].set(xlabel=r'$t$' + ' (s)')
        ax[0].grid(linestyle='--')
        ax[1].grid(linestyle='--')
        ax[0].tick_params(axis='y', which='major', pad=2)

        ax[0].sharex(ax[1])
        ax[0].tick_params(labelbottom=False)

        handles, labels = axs['traj'].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            bbox_to_anchor=(0.5, 0.03),
            loc='upper center',
            ncol=4,
        )
        fig.text(0.034, 1.015, r'$(a)$')
        fig.text(0.56, 1.015, r'$(b)$')
        fig_dict["F_{}".format(i)] = fig

    fig_dict['F_{}'.format(val)].savefig(
        'build/figures/paper/{}_trajectory_err_{}.png'.format(robot, variance),
        bbox_inches='tight')

    fig_dict['F_{}'.format(val)].savefig(
        'build/figures/paper/{}_trajectory_err_{}.pdf'.format(robot, variance),
        bbox_inches='tight')


def plot_centers(
    true_val_data: np.ndarray,
    norm_params: np.ndarray,
    robot: str,
    val: int = 2,
    n: int = None,
    **kwargs,
) -> None:

    n_val_eps = int(np.max(true_val_data[:, 0]) + 1)

    plt.rcParams.update(**kwargs)
    usetex = True if shutil.which('latex') else False
    if usetex:
        plt.rc('text', usetex=True)

    fig_dict = {}

    conv_param = 2.54 if robot == 'soft_robot' else 1

    x = true_val_data[:, :3]
    data = x[x[:, 0] == val][:n, 1:]
    centers = pykoop.QmcCenters(n_centers=10, random_state=1)
    centers.fit(data)

    r_max = 2
    b = 0.00001
    shape = 0.5
    rad = shape * np.linspace(0, r_max, 100) + b
    theta = np.linspace(0, 2 * np.pi, 100)
    r, th = np.meshgrid(rad, theta)
    # z = r**2 * np.log(r)
    z = np.exp(-r**2)

    fig, ax = plt.subplots(
        figsize=(3, 3),
        layout='constrained',
    )
    # levels = matplotlib.ticker.MaxNLocator(nbins=15).tick_values(z.min(), z.max())
    # norm = matplotlib.colors.BoundaryNorm(levels, ncolors=5, clip=True)

    for i in range(centers.centers_.shape[0]):
        Center = centers.centers_[i, :]

        axin = ax.inset_axes([Center[0] - 0.06, Center[1] - 0.1, 0.2, 0.2],
                             polar=True)
        t = axin.pcolormesh(th,
                            r,
                            z,
                            edgecolors='none',
                            cmap='Reds',
                            shading='nearest')
        axin.patch.set_alpha(0.01)
        # axin.plot(theta, r, color='k', linestyle='--')
        # axin.grid()
        axin.set_xticks([])
        axin.set_yticks([])
        axin.set_axis_off()

    ax.plot(true_val_data[true_val_data[:, 0] == val, 1][:n] *
            (conv_param * norm_params[0]),
            true_val_data[true_val_data[:, 0] == val, 2][:n] *
            (conv_param * norm_params[1]),
            label='Ground truth',
            color='k',
            linestyle='-',
            zorder=12)

    if val == val:

        ax.scatter(true_val_data[true_val_data[:, 0] == val, 1][0] *
                   (conv_param * norm_params[0]),
                   true_val_data[true_val_data[:, 0] == val, 2][0] *
                   (conv_param * norm_params[1]),
                   marker='x',
                   s=25,
                   color='k',
                   zorder=14)

        ax.scatter(true_val_data[true_val_data[:, 0] == val, 1][n - 1] *
                   (conv_param * norm_params[0]),
                   true_val_data[true_val_data[:, 0] == val, 2][n - 1] *
                   (conv_param * norm_params[1]),
                   marker='x',
                   s=25,
                   color='k',
                   zorder=14)

        # ax.scatter(centers.centers_[:, 0] * (conv_param * norm_params[0]),
        #            centers.centers_[:, 1] * (conv_param * norm_params[1]),
        #            marker='.',
        #            s=25,
        #            color='k',
        #            zorder=14)

        if robot == 'soft_robot':
            ax.set(ylabel=r'$x_{}$ (cm)'.format(2))
            ax.set(xlabel=r'$x_{}$ (cm)'.format(1))
        else:
            ax.set(ylabel=r'$x_{}$ (mm)'.format(2))
            ax.set(xlabel=r'$x_{}$ (mm)'.format(1))
        # ax.set_ylim(min_y - np.abs(min_y / 10), max_y + np.abs(max_y / 10))
        # ax.set_xlim(min_x - np.abs(min_x / 10), max_x + np.abs(max_x / 10))
        if robot == 'soft_robot':
            if val == 3:
                ax.set_ylim(0, 20)
                ax.set_xlim(0, 20)
                ax.set_xticks(np.linspace(0, 20, 11))
                ax.set_yticks(np.linspace(0, 20, 11))
            if val == 2:
                ax.set_ylim(0, 20)
                ax.set_xlim(0, 20)
                ax.set_xticks(np.linspace(0, 20, 11))
                ax.set_yticks(np.linspace(0, 20, 11))
            if val == 1:
                ax.set_xlim(-2, 10)
                ax.set_ylim(-14, -2)
                ax.set_xticks(np.linspace(-2, 10, 7))
                ax.set_yticks(np.linspace(-14, -2, 7))
            if val == 0:
                ax.set_xlim(-16, -6)
                ax.set_ylim(-6, 6)
                ax.set_xticks(np.linspace(-16, -6, 6))
                ax.set_yticks(np.linspace(-6, 6, 7))
        elif robot == 'nl_msd':
            if val == 0:
                ax.set_ylim(-4, 6)
                ax.set_xlim(-4, 6)
                ax.set_xticks(np.linspace(-4, 6, 6))
                ax.set_yticks(np.linspace(-4, 6, 6))
            if val == 1:
                ax.set_ylim(-6, 6)
                ax.set_xlim(-6, 6)
                ax.set_xticks(np.linspace(-6, 6, 7))
                ax.set_yticks(np.linspace(-6, 6, 7))

        ax.set_aspect('equal')

        ax.grid(linestyle='--')
        ax.tick_params(axis='y', which='major', pad=2)
        # fig.colorbar(t, location='right', ax=ax, fraction=0.05)
        # ax.tick_params(labelbottom=False)

        # handles, labels = axs['traj'].get_legend_handles_labels()
        # fig.legend(
        #     handles,
        #     labels,
        #     bbox_to_anchor=(0.5, 0.03),
        #     loc='upper center',
        #     ncol=4,
        # )
        fig_dict["F_{}".format(val)] = fig

    fig_dict['F_{}'.format(val)].savefig(
        'build/figures/paper/{}_centers.png'.format(robot),
        bbox_inches='tight')

    fig_dict['F_{}'.format(val)].savefig(
        'build/figures/paper/{}_centers.pdf'.format(robot),
        bbox_inches='tight')

    test = 1


def plot_frob_err(frob_error: Dict[str, np.ndarray], variances: np.ndarray,
                  snr: np.ndarray, path: str, **kwargs) -> None:

    plt.rcParams.update(**kwargs)
    usetex = True if shutil.which('latex') else False
    if usetex:
        plt.rc('text', usetex=True)
    i = 0

    fig, axs = plt.subplots(1, 3, layout='constrained', figsize=(5.3348, 2))
    # for (k,v), (k2,v2) in zip(d.items(), d2.items()):
    for (matrix, err), ax in zip(frob_error.items(), range(len(axs))):
        for tag, data in err.items():
            axs[ax].plot(snr,
                         data,
                         color=color_dict[tag],
                         label=tag,
                         linestyle=linestyle_dict[tag])

        if usetex == True:
            axs[ax].set(
                ylabel=
                r'$\frac{{\left\|\mathbf{{{}}}_\mathrm{{true}} - \bf{{{}}}_\mathrm{{approx}}\right\|_\ensuremath{{\mathsf{{F}}}}}}{{\left\|\mathbf{{{}}}_\mathrm{{true}}\right\|_\ensuremath{{\mathsf{{F}}}}}}$'
                .format(matrix, matrix, matrix))
        else:
            axs[ax].set(ylabel='rel. frob. norm of {}'.format(matrix))
        axs[ax].grid(linestyle='--')
        # axs[ax].set_xlim(np.min(snr), np.max(snr))
        # axs[ax].set_xticks(np.arange(10, 60, 10))
        axs[ax].set(xlabel=r'SNR')

    # axs[0].set_yticks(np.linspace(0, 0.1, 6))
    axs[1].sharey(axs[0])
    axs[1].tick_params(labelleft=False)
    # axs[2].set_yticks(np.linspace(0, 0.01, 6))
    axs[0].set_yticks(np.linspace(0, 2.5, 6))
    axs[2].set_yticks(np.linspace(0, 5, 6))

    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.5, 0.04),
        loc='upper center',
        ncol=3,
    )

    fig.savefig('{}/frob_norm_sqrd.png'.format(path), bbox_inches='tight')
    fig.savefig('{}/frob_norm_sqrd.pdf'.format(path), bbox_inches='tight')


def plot_new_eps(**kwargs) -> None:

    plt.rcParams.update(**kwargs)
    usetex = True if shutil.which('latex') else False
    if usetex:
        plt.rc('text', usetex=True)
    i = 0

    new_eps = [
        1e-04, 5e-04, 0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5,
        0.7, 1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 82.96229494436959, 100, 200,
        300, 500, 700, 1000, 2000, 3000, 5000, 7000, 10000
    ]
    P_val = np.zeros(len(new_eps))
    diff_val = np.zeros(len(new_eps))
    i = 0

    for eps in new_eps:
        path = 'build/minimize_val/{}.bin'.format(eps)
        with open(path, 'rb') as f:
            min_val = pickle.load(f)
        P_val[i] = min_val[str(eps)][0]
        diff_val[i] = min_val[str(eps)][1]
        i = i + 1

    fig, ax = plt.subplots(layout='constrained', figsize=(5.3348, 3))

    ax.plot(new_eps,
            P_val / np.max(P_val),
            color=color_dict['FBEDMD'],
            label=r'$\left\|\mathbf{P}\right\|_\ensuremath{\mathsf{F}}$')
    ax.plot(
        new_eps,
        diff_val / np.max(diff_val),
        color=color_dict['EDMD'],
        label=
        r'$\left\|\mathbf{G}_\mathrm{f}\mathbf{H}_\mathrm{f}^\dagger - [\mathbf{A}_\mathrm{ff} \;\; \mathbf{B}_\mathrm{ff}]\right\|_\ensuremath{\mathsf{F}}$'
    )
    # \begin{bmatrix} \mathbf{A}_\mathrm{ff} & \mathbf{B}_\mathrm{ff} \end{bmatrix}
    ax.plot([82.96229494436959, 82.96229494436959], [0, 1],
            color='k',
            linestyle='--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(linestyle='--')
    ax.set_xlim(np.min(new_eps), np.max(new_eps))
    ax.set_ylim(0, 1)
    # ax.set_xticks(np.arange(10, 60, 10))
    ax.set(xlabel=r'$\epsilon$')
    ax.set(ylabel=r'Cost function value')
    # ax.set_yticks(np.linspace(0, 2.5, 6))

    # handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(
        bbox_to_anchor=(0.46, 0.31),
        loc='upper center',
        ncol=4,
    )

    path = 'build/figures/paper'
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fig.savefig('{}/new_eps.png'.format(path), bbox_inches='tight')
    fig.savefig('{}/new_eps.pdf'.format(path), bbox_inches='tight')


def plot_polar(koop_matrices: Dict[str, np.ndarray], path: str, robot: str,
               **kwargs):

    plt.rcParams.update(**kwargs)
    usetex = True if shutil.which('latex') else False
    if usetex:
        plt.rc('text', usetex=True)

    p_theta, p = koop_matrices[list(koop_matrices.keys())[0]].shape
    n_inputs = p - p_theta

    fig = plt.figure(figsize=(5.3348, 3.5), constrained_layout=True)
    ax = plt.subplot(projection='polar')
    ax.set_xlabel(r'$\mathrm{Re}(\lambda)$')
    ax.set_ylabel(r'$\mathrm{Im}(\lambda)$', labelpad=30)

    theta_min = -12
    theta_max = 12

    axin = ax.inset_axes([1.05, -0.3, 0.7, 0.7], projection='polar')

    axes = [ax, axin]
    sub_eig = 0
    sup_eig = 2
    max_eig = 0
    i = 0

    koop_matrices_new = {}
    koop_matrices_new['EDMD'] = koop_matrices['EDMD']
    koop_matrices_new['EDMD-AS'] = koop_matrices['EDMD-AS']
    koop_matrices_new['FBEDMD'] = koop_matrices['FBEDMD']
    koop_matrices_new['FBEDMD-AS'] = koop_matrices['FBEDMD-AS']
    koop_matrices_new['TEDMD'] = koop_matrices['TEDMD']
    koop_matrices_new['TEDMD-AS'] = koop_matrices['TEDMD-AS']

    zorder = {}
    zorder['EDMD'] = 7
    zorder['EDMD-AS'] = 8
    zorder['FBEDMD'] = 5
    zorder['FBEDMD-AS'] = 6
    zorder['TEDMD'] = 9
    zorder['TEDMD-AS'] = 10

    markers = ['P', 'X', 's', 'D', 'v', 'o'
               ] + ['o'] * (len(koop_matrices.keys()) - 6)

    # plot eigenvalues
    for tag, U in koop_matrices_new.items():
        eigv = scipy.linalg.eig(U[:, :-n_inputs])[0]
        marker = markers.pop()

        for axx in axes:
            axx.scatter(np.angle(eigv),
                        np.absolute(eigv),
                        zorder=zorder[tag],
                        marker=marker,
                        s=60,
                        color=color_dict[tag],
                        linewidths=0.5,
                        edgecolors='w',
                        label=tag)
        i += 1

    unit_angles = np.linspace(0, 2 * np.pi, 100)
    unit_radius = np.ones_like(unit_angles)
    axin.plot(unit_angles, unit_radius, color='k', linestyle='--', linewidth=1)

    axin.set_thetamin(theta_min)
    axin.set_thetamax(theta_max)
    if robot == 'soft_robot':
        sub = 0.9995
        sup = 1.0004
    else:
        sub = 0.9998
        sup = 1.00005

    # sub = float("{:.5f}".format(sub))
    # sup = float("{:.5f}".format(sup))
    # axin.set_rticks([sub - (1 - sub), sub, 1.0, sup])
    # axin.set_rmin(sub - 0.1)
    # axin.set_rmax(sup + 0.1)
    # axin.set_yticks(sub, sup)
    # axin.set_yticklabels(['1e-2', '1e2'])
    axin.set_xticks([
        theta_min * np.pi / 180, theta_min / 2 * np.pi / 180, 0,
        theta_max / 2 * np.pi / 180, theta_max * np.pi / 180
    ])

    ax.plot(unit_angles, unit_radius, color='k', linestyle='--', linewidth=1)

    ax.set_rlim(0, 1.5)
    ax.set_rticks([0, 0.5, 1, 1.5])
    ax.grid(linestyle='--')
    axin.set_yticks([sub, sup])
    axin.set_yticklabels([str(sub), str(sup)])
    axin.set_rmin(sub)
    axin.set_rmax(sup)
    axin.grid(linestyle='--')

    # Create lines linking border to zoomed plot
    axin.annotate(
        '',
        xy=(0, sup),
        xycoords=ax.transData,
        xytext=(theta_max * np.pi / 180, sup),
        textcoords=axin.transData,
        arrowprops={
            'arrowstyle': '-',
            'color': 'k',
            'shrinkA': 0,
            'shrinkB': 0,
        },
    )

    axin.annotate(
        '',
        xy=(0, sub),
        xycoords=ax.transData,
        xytext=(0, sub),
        textcoords=axin.transData,
        arrowprops={
            'arrowstyle': '-',
            'color': 'k',
            'shrinkA': 0,
            'shrinkB': 0,
        },
    )

    axin.set_zorder(15)

    ax.legend(
        bbox_to_anchor=(0.74, -0.2),
        loc='upper center',
        ncol=3,
    )

    fig.savefig('build/figures/paper/{}_polar.pdf'.format(robot))
    fig.savefig('build/figures/paper/{}_polar.png'.format(robot))


def plot_polar_const(koop_matrices: Dict[str, np.ndarray], path: str,
                     robot: str, **kwargs):

    plt.rcParams.update(**kwargs)
    usetex = True if shutil.which('latex') else False
    if usetex:
        plt.rc('text', usetex=True)

    p_theta, p = koop_matrices[list(koop_matrices.keys())[0]].shape
    n_inputs = p - p_theta

    fig = plt.figure(figsize=(5.3348, 3.5), constrained_layout=True)
    ax = plt.subplot(projection='polar')
    ax.set_xlabel(r'$\mathrm{Re}(\lambda)$')
    ax.set_ylabel(r'$\mathrm{Im}(\lambda)$', labelpad=30)

    theta_min = -12
    theta_max = 12

    axin = ax.inset_axes([1.05, -0.3, 0.7, 0.7], projection='polar')

    axes = [ax, axin]
    sub_eig = 0
    sup_eig = 2
    max_eig = 0
    i = 0

    with open("build/others/matrices.bin", "rb") as f:
        mat_data = pickle.load(f)

    mat = {}
    mat = mat_data
    mat['EDMD'] = koop_matrices['EDMD'][:, :-n_inputs]

    mat_new = {}
    mat_new['EDMD'] = mat['EDMD']
    mat_new['Af'] = mat['Af']
    mat_new['Ab'] = mat['Ab']
    mat_new['A'] = mat['A']

    zorder = {}
    zorder['EDMD'] = 8
    zorder['A'] = 7
    zorder['Af'] = 9
    zorder['Ab'] = 6

    tags = {}
    tags['EDMD'] = r'$\mathbf{A}_\mathrm{EDMD}$'
    tags['A'] = r'$\tilde{\mathbf{A}}$'
    tags['Af'] = r'$\mathbf{A}_\mathrm{ff}$'
    tags['Ab'] = r'$\mathbf{A}_\mathrm{bb}$'

    markers = ['s', 'D', 'v', 'o'] + ['o'] * (len(koop_matrices.keys()) - 4)

    # plot eigenvalues
    for tag, A in mat_new.items():
        eigv = scipy.linalg.eig(mat[tag])[0]
        eigv = np.sort(eigv)
        if tag == 'Ab':
            eigv = eigv[:3]
        else:
            eigv = eigv[-3:]

        marker = markers.pop()

        for axx in axes:
            axx.scatter(np.angle(eigv),
                        np.absolute(eigv),
                        zorder=zorder[tag],
                        marker=marker,
                        s=60,
                        color=color_dict2[tag],
                        linewidths=0.5,
                        edgecolors='w',
                        label=tags[tag])
        i += 1

    unit_angles = np.linspace(0, 2 * np.pi, 100)
    unit_radius = np.ones_like(unit_angles)
    axin.plot(unit_angles, unit_radius, color='k', linestyle='--', linewidth=1)

    axin.set_thetamin(theta_min)
    axin.set_thetamax(theta_max)

    sub = 0.995
    sup = 1.001

    sub = float("{:.5f}".format(sub))
    sup = float("{:.5f}".format(sup))
    axin.set_rticks([sub - (1 - sub), sub, 1.0, sup])
    axin.set_rmin(sub)
    axin.set_rmax(sup)
    axin.set_xticks([
        theta_min * np.pi / 180, theta_min / 2 * np.pi / 180, 0,
        theta_max / 2 * np.pi / 180, theta_max * np.pi / 180
    ])

    ax.plot(unit_angles, unit_radius, color='k', linestyle='--', linewidth=1)

    ax.set_rlim(0, 1.5)
    ax.set_rticks([0, 0.5, 1, 1.5])
    ax.grid(linestyle='--')
    axin.set_rticks([sub, sup])
    axin.set_rmin(sub)
    axin.set_rmax(sup)
    axin.grid(linestyle='--')

    # Create lines linking border to zoomed plot
    axin.annotate(
        '',
        xy=(0, sup),
        xycoords=ax.transData,
        xytext=(theta_max * np.pi / 180, sup),
        textcoords=axin.transData,
        arrowprops={
            'arrowstyle': '-',
            'color': 'k',
            'shrinkA': 0,
            'shrinkB': 0,
        },
    )

    axin.annotate(
        '',
        xy=(0, sub),
        xycoords=ax.transData,
        xytext=(0, sub),
        textcoords=axin.transData,
        arrowprops={
            'arrowstyle': '-',
            'color': 'k',
            'shrinkA': 0,
            'shrinkB': 0,
        },
    )

    axin.set_zorder(15)

    ax.legend(
        bbox_to_anchor=(0.74, -0.2),
        loc='upper center',
        ncol=4,
    )

    fig.savefig('build/figures/paper/polar_const.pdf')
    fig.savefig('build/figures/paper/polar_const.png')


def summary_fig(
    val_data: Dict[str, np.ndarray],
    true_val_data: np.ndarray,
    path: str,
    norm_params: np.ndarray,
    robot: str,
    val: int = 2,
    n: int = None,
    **kwargs,
) -> None:

    if n is None:
        n = val_data[list(val_data.keys())[0]].shape[0]

    n_val_eps = int(np.max(true_val_data[:, 0]) + 1)

    plt.rcParams.update(**kwargs)
    plt.rc('font', size=12)
    usetex = True if shutil.which('latex') else False
    if usetex:
        plt.rc('text', usetex=True)

    fig_dict = {}

    conv_param = 2.54 if robot == 'soft_robot' else 1

    for i in range(n_val_eps):
        fig, ax = plt.subplots(
            figsize=(5, 2.2),
            layout='constrained',
        )
        max_y = 0
        min_y = 0
        max_x = 0
        min_x = 0

        ax.plot(true_val_data[true_val_data[:, 0] == i, 1][:(n - 1000)] *
                (conv_param * norm_params[0]),
                true_val_data[true_val_data[:, 0] == i, 2][:(n - 1000)] *
                (conv_param * norm_params[1]),
                label='Ground truth',
                color='k',
                linestyle=':',
                zorder=5,
                linewidth=2.5)

        for tag, data in val_data.items():

            if np.isnan(data[data[:, 0] == i,
                             1][:(n - 1000)]).any() or np.isnan(
                                 data[data[:, 0] == i, 2][:(n - 1000)]).any():
                print('NaN detected in {}.'.format(tag))
                continue
            else:
                if tag == 'EDMD' or tag == 'FBEDMD':
                    continue
                else:
                    if tag == "EDMD-AS":
                        ax.plot(data[data[:, 0] == i, 1][:(n - 1000)] *
                                (conv_param * norm_params[0]),
                                data[data[:, 0] == i, 2][:(n - 1000)] *
                                (conv_param * norm_params[1]),
                                label=tag + " (biased)",
                                color=color_list[2],
                                linestyle=linestyle_dict['EDMD'],
                                zorder=4,
                                linewidth=2.5)
                    else:
                        ax.plot(data[data[:, 0] == i, 1][:(n - 1000)] *
                                (conv_param * norm_params[0]),
                                data[data[:, 0] == i, 2][:(n - 1000)] *
                                (conv_param * norm_params[1]),
                                label=tag + " (unbiased)",
                                color=color_list[1],
                                linestyle=linestyle_dict['EDMD'],
                                zorder=3,
                                linewidth=2.5)
            temp_x = data[data[:, 0] == i,
                          1][:n] * (conv_param * norm_params[0])
            temp_y = data[data[:, 0] == i,
                          2][:n] * (conv_param * norm_params[1])
            max_x = np.max(temp_x) if (np.max(temp_x) > max_x) else max_x
            min_x = np.min(temp_x) if (np.min(temp_x) < min_x
                                       and np.min(temp_x) < 0) else min_x
            max_y = np.max(temp_y) if (np.max(temp_y) > max_y) else max_y
            min_y = np.min(temp_y) if (np.min(temp_y) < min_y
                                       and np.min(temp_y) < 0) else min_y

        if robot == 'soft_robot':
            ax.set(ylabel=r'$x_{}$ (cm)'.format(2))
            ax.set(xlabel=r'$x_{}$ (cm)'.format(1))
        else:
            ax.set(ylabel=r'$x_{}$'.format(2))
            ax.set(xlabel=r'$x_{}$'.format(1))
        if robot == 'soft_robot':
            ax.set_ylim(0, 20)
            ax.set_xlim(0, 20)
            ax.set_xticks(np.linspace(0, 20, 11))
            ax.set_yticks(np.linspace(0, 20, 11))

        ax.set_aspect('equal')

        ax.tick_params(labelleft=False, labelbottom=False)
        ax.tick_params(axis='both',
                       which='both',
                       bottom=False,
                       top=False,
                       labelbottom=False,
                       left=False,
                       labelleft=False)

        fig.text(0.49, 0.8, r'$\mathbf{U}_\mathrm{f}$')
        fig.text(0.595, 0.8, r'$\tilde{\mathbf{U}}$')

        handles, labels = ax.get_legend_handles_labels()

        fig_dict["F_{}".format(i)] = fig

    fig_dict['F_{}'.format(val)].savefig(
        'build/figures/paper/{}_summary_trajectory.png'.format(robot),
        bbox_inches='tight')

    fig_dict['F_{}'.format(val)].savefig(
        'build/figures/paper/{}_summary_trajectory.pdf'.format(robot),
        bbox_inches='tight')


def print_koop_matrices(koop_matrices: Dict[str, np.ndarray], **kwargs):

    n_inputs = koop_matrices[list(
        koop_matrices.keys())[0]].shape[1] - koop_matrices[list(
            koop_matrices.keys())[0]].shape[0]

    for tag, U in koop_matrices.items():
        print('Eigenvalues of A_{}: '.format(tag),
              scipy.linalg.eig(U[:, :-n_inputs])[0])


class LmiEdmd(LmiRegressor):
    """LMI-based EDMD with regularization. Inspired from the pykoop package.

    """

    def __init__(
        self,
        alpha: float = 0,
        ratio: float = 1,
        reg_method: str = 'tikhonov',
        inv_method: str = 'svd',
        tsvd: Optional[tsvd.Tsvd] = None,
        square_norm: bool = False,
        picos_eps: float = 0,
        solver_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Instantiate :class:`LmiEdmd`. Inspired from the pykoop package.

        """
        self.alpha = alpha
        self.ratio = ratio
        self.reg_method = reg_method
        self.inv_method = inv_method
        self.tsvd = tsvd
        self.square_norm = square_norm
        self.picos_eps = picos_eps
        self.solver_params = solver_params

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        # Set solver parameters
        self.solver_params_ = self._default_solver_params.copy()
        if self.solver_params is not None:
            self.solver_params_.update(self.solver_params)
        # Compute regularization coefficients
        if self.reg_method == 'tikhonov':
            self.alpha_tikhonov_ = self.alpha
            self.alpha_other_ = 0.0
        else:
            self.alpha_tikhonov_ = self.alpha * (1.0 - self.ratio)
            self.alpha_other_ = self.alpha * self.ratio
        # Clone TSVD
        self.tsvd_ = (sklearn.base.clone(self.tsvd)
                      if self.tsvd is not None else tsvd.Tsvd())
        # Form optimization problem. Regularization coefficients must be scaled
        # because of how G and H are defined.
        q = X_unshifted.shape[0]
        problem = self._create_base_problem(X_unshifted, X_shifted,
                                            self.alpha_tikhonov_ / q,
                                            self.inv_method, self.tsvd_,
                                            self.picos_eps)
        if self.reg_method == 'twonorm':
            problem = lmi_regressors._add_twonorm(problem,
                                                  problem.variables['U'],
                                                  self.alpha_other_ / q,
                                                  self.square_norm,
                                                  self.picos_eps)
        elif self.reg_method == 'nuclear':
            problem = lmi_regressors._add_nuclear(problem,
                                                  problem.variables['U'],
                                                  self.alpha_other_ / q,
                                                  self.square_norm,
                                                  self.picos_eps)
        # Solve optimization problem
        problem.solve(**self.solver_params_)
        # Save solution status
        self.solution_status_ = problem.last_solution.claimedStatus
        # Extract solution from ``Problem`` object
        coef = self._extract_solution(problem)
        return coef

    def _validate_parameters(self) -> None:
        # Check regularization methods
        valid_reg_methods = ['tikhonov', 'twonorm', 'nuclear']
        if self.reg_method not in valid_reg_methods:
            raise ValueError('`reg_method` must be one of '
                             f'{valid_reg_methods}.')
        # Check ratio
        if (self.ratio <= 0) or (self.ratio > 1):
            raise ValueError('`ratio` must be positive and less than one.')

    @staticmethod
    def _create_base_problem(
        X_unshifted: np.ndarray,
        X_shifted: np.ndarray,
        alpha_tikhonov: float,
        inv_method: str,
        tsvd: tsvd.Tsvd,
        picos_eps: float,
    ) -> picos.Problem:
        """Create optimization problem."""
        # Validate ``alpha``
        if alpha_tikhonov < 0:
            raise ValueError('Parameter `alpha` must be positive or zero.')
        # Validate ``inv_method``
        valid_inv_methods = [
            'inv', 'pinv', 'eig', 'ldl', 'chol', 'sqrt', 'svd'
        ]
        if inv_method not in valid_inv_methods:
            raise ValueError('`inv_method` must be one of '
                             f'{valid_inv_methods}.')
        # Validate ``picos_eps``
        if picos_eps < 0:
            raise ValueError('Parameter `picos_eps` must be positive or zero.')
        var = np.var(X_shifted, axis=0)
        W = np.diag(1 / var)
        W = np.eye(W.shape[0])
        # Compute ``G`` and ``H``. ``alpha_tikhonov`` must already be scaled
        # by ``q`` if applicable.
        c, G, H, _ = lmi_regressors._calc_c_G_H(X_unshifted, X_shifted @ W.T,
                                                alpha_tikhonov)
        # Optimization problem
        problem = picos.Problem()
        # Constants
        G_T = picos.Constant('G^T', G.T)
        W = picos.Constant('W', W)
        q = X_shifted.shape[0]
        gamma = 10
        # Variables
        U = picos.RealVariable('U', (G.shape[0], H.shape[0]))
        Z = picos.SymmetricVariable('Z', (G.shape[0], G.shape[0]))
        # W = picos.RealVariable('W', (G.shape[0], G.shape[0]))
        v = picos.RealVariable('v', 1)
        Q = picos.SymmetricVariable('Q', (G.shape[1], G.shape[1]))
        # Constraints
        problem.add_constraint(Z >> picos_eps)
        # problem.add_constraint(Q >> picos_eps)
        # Choose method to handle inverse of H
        if inv_method == 'inv':
            H_inv = picos.Constant('H^-1', lmi_regressors._calc_Hinv(H))
            problem.add_constraint(
                picos.block([
                    [Z, U],
                    [U.T, H_inv],
                ]) >> picos_eps)
        elif inv_method == 'pinv':
            H_inv = picos.Constant('H^+', lmi_regressors._calc_Hpinv(H))
            problem.add_constraint(
                picos.block([
                    [Z, U],
                    [U.T, H_inv],
                ]) >> picos_eps)
        elif inv_method == 'eig':
            VsqrtLmb = picos.Constant('(V Lambda^(1/2))',
                                      lmi_regressors._calc_VsqrtLmb(H))
            problem.add_constraint(
                picos.block([
                    [Z, U * VsqrtLmb],
                    [VsqrtLmb.T * U.T, 'I'],
                ]) >> picos_eps)
        elif inv_method == 'ldl':
            LsqrtD = picos.Constant('(L D^(1/2))',
                                    lmi_regressors._calc_LsqrtD(H))
            problem.add_constraint(
                picos.block([
                    [Z, U * LsqrtD],
                    [LsqrtD.T * U.T, 'I'],
                ]) >> picos_eps)
        elif inv_method == 'chol':
            L = picos.Constant('L', lmi_regressors._calc_L(H))
            problem.add_constraint(
                picos.block([
                    [Z, U * L],
                    [L.T * U.T, 'I'],
                ]) >> picos_eps)
        elif inv_method == 'sqrt':
            sqrtH = picos.Constant('sqrt(H)', lmi_regressors._calc_sqrtH(H))
            problem.add_constraint(
                picos.block([
                    [Z, U * sqrtH],
                    [sqrtH.T * U.T, 'I'],
                ]) >> picos_eps)
        elif inv_method == 'svd':
            QSig = picos.Constant(
                'Q Sigma',
                lmi_regressors._calc_QSig(X_unshifted, alpha_tikhonov, tsvd))
            problem.add_constraint(
                picos.block([
                    [Z, W * U * QSig],
                    [QSig.T * U.T * W.T, 'I'],
                ]) >> picos_eps)
        else:
            # Should never, ever get here.
            assert False
        # Set objective
        obj = c - 2 * picos.trace(W * U * G_T * W.T) + picos.trace(Z)
        problem.set_objective('min', obj)
        return problem

    @staticmethod
    def _create_new_forw_base_problem(
        # Variablesm(
        X_unshifted: np.ndarray,
        X_shifted: np.ndarray,
        alpha_tikhonov: float,
        spectral_radius: float,
        picos_eps: float,
        new_eps: np.ndarray,
    ) -> picos.Problem:
        """Create optimization problem."""
        # Validate ``alpha``
        if alpha_tikhonov < 0:
            raise ValueError('Parameter `alpha` must be positive or zero.')
        # Validate ``inv_method``

        # Validate ``picos_eps``
        if picos_eps < 0:
            raise ValueError('Parameter `picos_eps` must be positive or zero.')
        # Compute ``G`` and ``H``. ``alpha_tikhonov`` must already be scaled
        # by ``q`` if applicable.
        # c, G, H, _ = lmi_regressors._calc_c_G_H(X_unshifted, X_shifted, alpha_tikhonov)
        # G = picos.Constant('G', G)
        # H = picos.Constant('H', H)

        # Optimization problem
        problem = picos.Problem()

        # Constants
        q = X_shifted.shape[0]
        p_theta = X_shifted.shape[1]
        Psi = X_unshifted.T
        Theta_plus = X_shifted.T
        n_inputs = Psi.shape[0] - Theta_plus.shape[0]

        _G = 1 / q * Theta_plus @ Psi.T
        # _G = _G + np.eye(_G.shape[0])*1e-5
        _H = 1 / q * Psi @ Psi.T
        # _H = _H + np.eye(_H.shape[0])*1e-5
        H_inv = scipy.linalg.pinv(_H)
        U_edmd = scipy.linalg.lstsq(_H.T, _G.T)[0].T
        # _R = scipy.linalg.cholesky(_Hb, lower=True)
        W = picos.Constant(
            'W',
            picos.block([[np.eye(p_theta),
                          np.zeros((p_theta, n_inputs))],
                         [np.zeros((n_inputs, p_theta)),
                          np.eye(n_inputs)]]))
        rho = picos.Constant('rho', spectral_radius)
        alpha = picos.Constant('alpha', 1)

        # Variables
        P = picos.SymmetricVariable('P', p_theta)
        B = picos.RealVariable('B', (_G.shape[0], _H.shape[0] - _G.shape[0]))
        K = picos.RealVariable('K', (p_theta, p_theta))
        P_tilde = picos.block(
            [[P, np.zeros((P.shape[0], H_inv.shape[0] - P.shape[0]))],
             [
                 np.zeros((H_inv.shape[0] - P.shape[0], P.shape[0])),
                 alpha * np.eye(H_inv.shape[0] - P.shape[0])
             ]])
        gamma = picos.RealVariable('gamma_f', 1)
        Z = picos.SymmetricVariable('Z', _H.shape[0])

        # Constraints for forward dynamics
        # problem.add_constraint(P >> picos_eps)
        problem.add_constraint(P >> new_eps)
        problem.add_constraint(
            picos.block([
                [rho * P, K],
                [K.T, rho * P],
            ]) >> picos_eps)
        problem.add_constraint(picos.trace(Z) << 1)
        problem.add_constraint(Z >> picos_eps)
        problem.add_constraint(
            picos.block([[Z, (U_edmd * P_tilde - picos.block([[K, B]])).T],
                         [(U_edmd * P_tilde - picos.block([[K, B]])), gamma *
                          np.eye(p_theta)]]) >> picos_eps)

        obj = gamma
        problem.set_objective('min', obj)
        return problem

    @staticmethod
    def _create_new_avg_base_problem(
        # Variablesm(
        X_unshifted: np.ndarray,
        X_shifted: np.ndarray,
        alpha_tikhonov: float,
        spectral_radius: float,
        picos_eps: float,
        new_eps: np.ndarray,
    ) -> picos.Problem:
        """Create optimization problem."""
        # Validate ``alpha``
        if alpha_tikhonov < 0:
            raise ValueError('Parameter `alpha` must be positive or zero.')
        # Validate ``inv_method``

        # Validate ``picos_eps``
        if picos_eps < 0:
            raise ValueError('Parameter `picos_eps` must be positive or zero.')
        # Compute ``G`` and ``H``. ``alpha_tikhonov`` must already be scaled
        # by ``q`` if applicable.
        # c, G, H, _ = lmi_regressors._calc_c_G_H(X_unshifted, X_shifted, alpha_tikhonov)
        # G = picos.Constant('G', G)
        # H = picos.Constant('H', H)

        # Optimization problem
        problem = picos.Problem()

        # Constants
        # var = np.var(X_shifted, axis=0)
        # W = np.diag(1 / var)
        W = np.eye(X_shifted.shape[1])
        q = X_shifted.shape[0]
        p_theta = X_shifted.shape[1]
        Psi_f = X_unshifted.T
        Theta_plus_f = X_shifted.T
        n_inputs = Psi_f.shape[0] - Theta_plus_f.shape[0]
        Psi_b = np.zeros(Psi_f.shape)
        Theta_plus_b = np.zeros(Theta_plus_f.shape)
        Psi_b[:-n_inputs, :] = Theta_plus_f
        Psi_b[-n_inputs:, :] = Psi_f[-n_inputs:, :]
        Theta_plus_b = Psi_f[:-n_inputs, :]

        _Gf = 1 / q * Theta_plus_f @ Psi_f.T
        # _Gf = _Gf + np.eye(_Gf.shape[0])*1e-5
        _Hf = 1 / q * Psi_f @ Psi_f.T
        # _Hf = _Hf + np.eye(_Hf.shape[0])*1e-5
        H_inv_f = scipy.linalg.pinv(_Hf)
        U_edmd_f = scipy.linalg.lstsq(_Hf.T, _Gf.T)[0].T
        _Gb = 1 / q * Theta_plus_b @ Psi_b.T
        _Hb = 1 / q * Psi_b @ Psi_b.T
        H_inv_b = scipy.linalg.pinv(_Hb)
        U_edmd_b = scipy.linalg.lstsq(_Hb.T, _Gb.T)[0].T
        # _R = scipy.linalg.cholesky(_Hb, lower=True)
        W = picos.Constant('W', W)
        rho = picos.Constant('rho', spectral_radius)
        alpha = picos.Constant('alpha', 1)

        # Variables
        P = picos.SymmetricVariable('P', p_theta)
        Bf = picos.RealVariable('Bf',
                                (_Gf.shape[0], _Hf.shape[0] - _Gf.shape[0]))
        Kf = picos.RealVariable('Kf', (p_theta, p_theta))
        Bb = picos.RealVariable('Bb',
                                (_Gb.shape[0], _Hb.shape[0] - _Gb.shape[0]))
        Kb = picos.RealVariable('Kb', (p_theta, p_theta))
        P_tilde = picos.block(
            [[P, np.zeros((P.shape[0], H_inv_f.shape[0] - P.shape[0]))],
             [
                 np.zeros((H_inv_f.shape[0] - P.shape[0], P.shape[0])),
                 alpha * np.eye(H_inv_f.shape[0] - P.shape[0])
             ]])
        gamma_f = picos.RealVariable('gamma_f', 1)
        gamma_b = picos.RealVariable('gamma_b', 1)
        Zf = picos.SymmetricVariable('Zf', _Hf.shape[0])
        Zb = picos.SymmetricVariable('Zb', _Hb.shape[0])
        Qf = picos.SymmetricVariable('Qf', _Gf.shape[0])

        # Constraints for forward dynamics
        # problem.add_constraint(P >> picos_eps)
        # if new_eps < 1:
        #     new_eps = 1
        problem.add_constraint(P >> new_eps)
        problem.add_constraint(
            picos.block([
                [rho * P, Kf],
                [Kf.T, rho * P],
            ]) >> picos_eps)
        problem.add_constraint(picos.trace(Zf) << 1)
        problem.add_constraint(Zf >> picos_eps)
        problem.add_constraint(
            picos.block([
                [Zf, (W * U_edmd_f * P_tilde - W * picos.block([[Kf, Bf]])).T],
                [(W * U_edmd_f * P_tilde -
                  W * picos.block([[Kf, Bf]])), gamma_f * np.eye(p_theta)]
            ]) >> picos_eps)

        # Constraints for backward dynamics
        problem.add_constraint(rho * Kb + rho * Kb.T - 2 * P >> picos_eps)
        problem.add_constraint(picos.trace(Zb) << 1)
        problem.add_constraint(Zb >> picos_eps)
        problem.add_constraint(
            picos.block([
                [Zb, (W * U_edmd_b * P_tilde - W * picos.block([[Kb, Bb]])).T],
                [(W * U_edmd_b * P_tilde -
                  W * picos.block([[Kb, Bb]])), gamma_b * np.eye(p_theta)]
            ]) >> picos_eps)

        obj = gamma_f + gamma_b
        problem.set_objective('min', obj)
        return problem

    @staticmethod
    def _create_tedmd_AS_problem(
        # Variablesm(
        X_unshifted: np.ndarray,
        X_shifted: np.ndarray,
        alpha_tikhonov: float,
        spectral_radius: float,
        picos_eps: float,
        new_eps: np.ndarray,
    ) -> picos.Problem:
        """Create optimization problem."""
        # Validate ``alpha``
        if alpha_tikhonov < 0:
            raise ValueError('Parameter `alpha` must be positive or zero.')
        # Validate ``inv_method``

        # Validate ``picos_eps``
        # if picos_eps < 0:
        #     raise ValueError('Parameter `picos_eps` must be positive or zero.')
        # Compute ``G`` and ``H``. ``alpha_tikhonov`` must already be scaled
        # by ``q`` if applicable.
        # c, G, H, _ = lmi_regressors._calc_c_G_H(X_unshifted, X_shifted, alpha_tikhonov)
        # G = picos.Constant('G', G)
        # H = picos.Constant('H', H)

        # Optimization problem
        problem = picos.Problem()

        # Constants
        q = X_shifted.shape[0]
        p_theta = X_shifted.shape[1]
        Psi = X_unshifted.T
        Theta_plus = X_shifted.T
        n_inputs = Psi.shape[0] - Theta_plus.shape[0]
        k = 0

        # kp = pykoop.Edmd(alpha=0.1)
        # kp.fit(Psi.T, Theta_plus.T)
        # U_edmd = kp.coef_.T

        _G = 1 / q * Theta_plus @ Psi.T
        # _G[:, :p_theta] + np.eye(p_theta) * k
        _H = 1 / q * Psi @ Psi.T
        _H_A = 1 / q * Psi[:-n_inputs, :] @ Psi[:-n_inputs, :].T
        _H_B = 1 / q * Psi[-n_inputs:, :] @ Psi[-n_inputs:, :].T
        _H_A = _H_A + np.eye(p_theta) * k
        H_inv = scipy.linalg.pinv(_H)
        H_inv_A = scipy.linalg.pinv(_H_A.T)
        U_edmd = scipy.linalg.lstsq(_H.T, _G.T)[0].T
        U_edmd_y = scipy.linalg.lstsq(_H_A.T, _G[:, :p_theta].T)[0].T

        # Obtain actual U_edmd from series expansion
        # for i in range(U_edmd_y.shape[1]):
        #     temp = np.eye(p_theta) @ (k * H_inv_A)**i @ U_edmd_y
        #     temp2 = temp if i == 0 else temp2 + temp
        # U_edmd = np.block([[temp2, U_edmd[:p_theta, p_theta:]]])

        # U_edmd = U_edmd + np.block([[np.eye(p_theta), U_edmd[:, -n_inputs:]]])

        # Variables
        A = picos.RealVariable('A', (p_theta, p_theta))
        B = picos.RealVariable('B', (p_theta, n_inputs))
        # K = picos.RealVariable('K', (p_theta, X_unshifted.shape[0]))
        # K2 = picos.RealVariable('K2', (p_theta, X_unshifted.shape[0]))
        # DX = picos.RealVariable('DX', (p_theta + n_inputs, X_shifted.shape[0]))
        DY = picos.RealVariable('DY', (p_theta, X_shifted.shape[0]))
        X_shifted = picos.Constant('X_shifted', X_shifted)
        X_unshifted = picos.Constant('X_unshifted', X_unshifted)
        # rho = picos.Constant('rho', spectral_radius)
        alpha = picos.Constant('alpha', 1)
        U_edmd = picos.Constant('U_edmd', U_edmd)

        # Variables
        P = picos.SymmetricVariable('P', p_theta)
        B = picos.RealVariable('B', (_G.shape[0], _H.shape[0] - _G.shape[0]))
        K = picos.RealVariable('K', (p_theta, p_theta))
        # P_tilde = P
        gamma = picos.RealVariable('gamma', 1)
        mu = picos.RealVariable('mu', 1)

        Z = picos.SymmetricVariable('Z', p_theta)
        Q = picos.SymmetricVariable('Q', n_inputs)

        # Constraints for forward dynamics
        # problem.add_constraint(P >> picos_eps)
        problem.add_constraint(P >> new_eps)
        rho = spectral_radius
        problem.add_constraint(
            picos.block([
                [rho * P, K],
                [K.T, rho * P],
            ]) >> picos_eps)
        problem.add_constraint(picos.trace(Z) << 1)
        problem.add_constraint(picos.trace(Q) << 1)
        problem.add_constraint(Z >> picos_eps)
        problem.add_constraint(Q >> picos_eps)
        problem.add_constraint(
            picos.block([[Z, ((U_edmd[:p_theta, :p_theta] * P) - K).T],
                         [(U_edmd[:p_theta, :p_theta] * P) - K, gamma *
                          np.eye(p_theta)]]) >> picos_eps)
        problem.add_constraint(
            picos.block([[Q, (
                U_edmd[:p_theta, p_theta:] -
                B).T], [U_edmd[:p_theta, p_theta:] - B, mu *
                        np.eye(p_theta)]]) >> picos_eps)

        obj = gamma + mu
        problem.set_objective('min', obj)
        return problem

        # Constraints for forward dynamics
        # problem.add_constraint(X_shifted.T +
        #                        DY == A * X_unshifted.T[:p_theta, :] +
        #                        B * X_unshifted.T[p_theta:, :] + K)
        # problem.add_constraint(X_shifted.T +
        #                        DY == A * X_unshifted.T[:p_theta, :] + K)
        # obj = gamma
        # obj = picos.Norm(picos.block([[K], [DY]]), p=2, q=2)
        # problem.set_objective('min', obj)
        # return problem

    @staticmethod
    def _create_Uedmd_problem(
        # Variablesm(
        X_unshifted: np.ndarray,
        X_shifted: np.ndarray,
        alpha_tikhonov: float,
        spectral_radius: float,
        picos_eps: float,
        new_eps: np.ndarray,
    ) -> picos.Problem:
        """Create optimization problem."""
        # Validate ``alpha``
        if alpha_tikhonov < 0:
            raise ValueError('Parameter `alpha` must be positive or zero.')
        # Validate ``inv_method``

        # Validate ``picos_eps``
        if picos_eps < 0:
            raise ValueError('Parameter `picos_eps` must be positive or zero.')
        # Compute ``G`` and ``H``. ``alpha_tikhonov`` must already be scaled
        # by ``q`` if applicable.
        # c, G, H, _ = lmi_regressors._calc_c_G_H(X_unshifted, X_shifted, alpha_tikhonov)
        # G = picos.Constant('G', G)
        # H = picos.Constant('H', H)

        # Optimization problem
        problem = picos.Problem()

        # Constants
        q = X_shifted.shape[0]
        p_theta = X_shifted.shape[1]
        Psi = X_unshifted.T
        Theta_plus = X_shifted.T
        n_inputs = Psi.shape[0] - Theta_plus.shape[0]

        _G = 1 / q * Theta_plus @ Psi.T
        # _G = _G + np.eye(_G.shape[0])*1e-5
        _H = 1 / q * Psi @ Psi.T
        # _H = _H + np.eye(_H.shape[0])*1e-5
        H_inv = scipy.linalg.pinv(_H)
        U_edmd = scipy.linalg.lstsq(_H.T, _G.T)[0].T
        # _R = scipy.linalg.cholesky(_Hb, lower=True)
        W = picos.Constant(
            'W',
            picos.block([[np.eye(p_theta),
                          np.zeros((p_theta, n_inputs))],
                         [np.zeros((n_inputs, p_theta)),
                          np.eye(n_inputs)]]))
        rho = picos.Constant('rho', spectral_radius)
        alpha = picos.Constant('alpha', 1)

        # Variables
        P = picos.SymmetricVariable('P', p_theta)
        B = picos.RealVariable('B', (_G.shape[0], _H.shape[0] - _G.shape[0]))
        K = picos.RealVariable('K', (p_theta, p_theta))
        P_tilde = picos.block(
            [[P, np.zeros((P.shape[0], H_inv.shape[0] - P.shape[0]))],
             [
                 np.zeros((H_inv.shape[0] - P.shape[0], P.shape[0])),
                 alpha * np.eye(H_inv.shape[0] - P.shape[0])
             ]])
        gamma = picos.RealVariable('gamma_f', 1)
        Z = picos.SymmetricVariable('Z', _H.shape[0])

        # Constraints for forward dynamics
        # problem.add_constraint(P >> picos_eps)
        problem.add_constraint(P >> new_eps)
        problem.add_constraint(
            picos.block([
                [rho * P, K],
                [K.T, rho * P],
            ]) >> picos_eps)
        problem.add_constraint(picos.trace(Z) << 1)
        problem.add_constraint(Z >> picos_eps)
        problem.add_constraint(
            picos.block([[Z, (U_edmd * P_tilde - picos.block([[K, B]])).T],
                         [(U_edmd * P_tilde - picos.block([[K, B]])), gamma *
                          np.eye(p_theta)]]) >> picos_eps)

        obj = gamma
        problem.set_objective('min', obj)
        return problem

    @staticmethod
    def _create_tedmd_AS_problem1(
        # Variablesm(
        X_unshifted: np.ndarray,
        X_shifted: np.ndarray,
        alpha_tikhonov: float,
        spectral_radius: float,
        picos_eps: float,
        new_eps: np.ndarray,
    ) -> picos.Problem:
        """Create optimization problem."""
        # Validate ``alpha``
        if alpha_tikhonov < 0:
            raise ValueError('Parameter `alpha` must be positive or zero.')
        # Validate ``inv_method``

        # Validate ``picos_eps``
        # if picos_eps < 0:
        #     raise ValueError('Parameter `picos_eps` must be positive or zero.')
        # Compute ``G`` and ``H``. ``alpha_tikhonov`` must already be scaled
        # by ``q`` if applicable.
        # c, G, H, _ = lmi_regressors._calc_c_G_H(X_unshifted, X_shifted, alpha_tikhonov)
        # G = picos.Constant('G', G)
        # H = picos.Constant('H', H)

        # Optimization problem
        problem = picos.Problem()

        # Constants
        q = X_shifted.shape[0]
        p_theta = X_shifted.shape[1]
        Psi = X_unshifted.T
        Theta_plus = X_shifted.T
        n_inputs = Psi.shape[0] - Theta_plus.shape[0]
        k = 1e1

        _G = 1 / q * Theta_plus @ Psi.T
        # _G[:, :p_theta] + np.eye(p_theta) * k
        _H = 1 / q * Psi @ Psi.T
        _H_A = 1 / q * Psi[:-n_inputs, :] @ Psi[:-n_inputs, :].T
        _H_B = 1 / q * Psi[-n_inputs:, :] @ Psi[-n_inputs:, :].T
        _H_A = _H_A + np.eye(p_theta) * k
        H_inv = scipy.linalg.pinv(_H)
        H_inv_A = scipy.linalg.pinv(_H_A)
        U_edmd = scipy.linalg.lstsq(_H.T, _G.T)[0].T
        # U_edmd_y = scipy.linalg.lstsq(_H_A.T, _G[:, :p_theta].T)[0].T

        # # Obtain actual U_edmd from series expansion
        # for i in range(U_edmd_y.shape[1]):
        #     temp = np.eye(p_theta) @ (k * H_inv_A)**i @ U_edmd_y
        #     temp2 = temp if i == 0 else temp2 + temp
        # U_edmd = np.block([[temp2, U_edmd[:p_theta, p_theta:]]])

        # U_edmd = U_edmd + np.block([[np.eye(15), U_edmd[:, -n_inputs:]]])

        # Variables
        A = picos.RealVariable('A', (p_theta, p_theta))
        B = picos.RealVariable('B', (p_theta, n_inputs))
        # K = picos.RealVariable('K', (p_theta, X_unshifted.shape[0]))
        # K2 = picos.RealVariable('K2', (p_theta, X_unshifted.shape[0]))
        # DX = picos.RealVariable('DX', (p_theta + n_inputs, X_shifted.shape[0]))
        DY = picos.RealVariable('DY', (p_theta, X_shifted.shape[0]))
        X_shifted = picos.Constant('X_shifted', X_shifted)
        X_unshifted = picos.Constant('X_unshifted', X_unshifted)
        # rho = picos.Constant('rho', spectral_radius)
        alpha = picos.Constant('alpha', 1)
        U_edmd = picos.Constant('U_edmd', U_edmd)

        # Variables
        P = picos.SymmetricVariable('P', p_theta)
        B = picos.RealVariable('B', (_G.shape[0], _H.shape[0] - _G.shape[0]))
        K = picos.RealVariable('K', (p_theta, p_theta))
        P_tilde = picos.block(
            [[P, np.zeros((P.shape[0], H_inv.shape[0] - P.shape[0]))],
             [
                 np.zeros((H_inv.shape[0] - P.shape[0], P.shape[0])),
                 alpha * np.eye(H_inv.shape[0] - P.shape[0])
             ]])
        # P_tilde = P
        gamma = picos.RealVariable('gamma_f', 1)
        Z = picos.SymmetricVariable('Z', _H.shape[0])

        # Constraints for forward dynamics
        # problem.add_constraint(P >> picos_eps)
        problem.add_constraint(P >> new_eps)
        rho = spectral_radius
        problem.add_constraint(
            picos.block([
                [rho * P, K],
                [K.T, rho * P],
            ]) >> picos_eps)
        # rho = rho + 0.1
        # problem.add_constraint(rho * K + rho * K.T - 2 * P >> picos_eps)
        problem.add_constraint(picos.trace(Z) << 1)
        problem.add_constraint(Z >> picos_eps)
        problem.add_constraint(
            picos.block([[Z, (U_edmd * P_tilde - picos.block([[K, B]])).T],
                         [(U_edmd * P_tilde - picos.block([[K, B]])), gamma *
                          np.eye(p_theta)]]) >> picos_eps)

        obj = gamma
        problem.set_objective('min', obj)
        return problem

        # Constraints for forward dynamics
        # problem.add_constraint(X_shifted.T +
        #                        DY == A * X_unshifted.T[:p_theta, :] +
        #                        B * X_unshifted.T[p_theta:, :] + K)
        # problem.add_constraint(X_shifted.T +
        #                        DY == A * X_unshifted.T[:p_theta, :] + K)
        # obj = gamma
        # obj = picos.Norm(picos.block([[K], [DY]]), p=2, q=2)
        # problem.set_objective('min', obj)
        # return problem


class LmiEdmdSpectralRadiusConstrForw(LmiRegressor):
    """LMI-based EDMD with spectral radius constraint. Inspired by the pykoop package.
    """

    def __init__(self,
                 spectral_radius: float = 1.0,
                 new_eps: np.ndarray = 1e-2,
                 max_iter: int = 100,
                 iter_atol: float = 1e-8,
                 iter_rtol: float = 0,
                 alpha: float = 0,
                 inv_method: str = 'svd',
                 tsvd: tsvd.Tsvd = None,
                 picos_eps: float = 1e-5,
                 P: np.ndarray = None,
                 solver_params: Dict[str, Any] = None) -> None:
        """Instantiate :class:`LmiEdmdSpectralRadiusConstr`. Inspired by the pykoop package.

        """
        self.spectral_radius = spectral_radius
        self.new_eps = new_eps
        self.max_iter = max_iter
        self.iter_atol = iter_atol
        self.iter_rtol = iter_rtol
        self.alpha = alpha
        self.inv_method = inv_method
        self.tsvd = tsvd
        self.picos_eps = picos_eps
        self.solver_params = solver_params
        self.P = P
        # self.cost = np.zeros((self.new_eps.shape))
        # self.new_rad = np.zeros((self.new_eps.shape))

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        # Set solver parameters
        self.solver_params_ = self._default_solver_params.copy()
        if self.solver_params is not None:
            self.solver_params_.update(self.solver_params)
        # Clone TSVD
        self.tsvd_ = (sklearn.base.clone(self.tsvd)
                      if self.tsvd is not None else tsvd.Tsvd())
        # Get needed sizes
        p = X_unshifted.shape[1]
        p_theta = X_shifted.shape[1]
        # Make initial guesses and iterate
        P = np.eye(p_theta)
        # Set scope of other variables
        U = np.zeros((p_theta, p))
        self.objective_log_ = []

        # Forw

        q = X_shifted.shape[0]
        p_theta = X_shifted.shape[1]
        Psi = X_unshifted.T
        Theta_plus = X_shifted.T
        n_inputs = Psi.shape[0] - Theta_plus.shape[0]
        new_eps_og = np.linalg.norm(Psi.T @ scipy.linalg.pinv(Psi @ Psi.T), 2)
        # new_eps_og = np.linalg.norm(
        #     Theta_plus @ Psi.T @ scipy.linalg.pinv(Psi @ Psi.T), 2)

        _G = 1 / q * Theta_plus @ Psi.T
        _H = 1 / q * Psi @ Psi.T
        H_inv = scipy.linalg.pinv(_H)
        U_edmd = scipy.linalg.lstsq(_H.T, _G.T)[0].T

        if self.new_eps == -1:
            self.new_eps = new_eps_og

        # self.solver_params_ = {
        #     'solver': 'mosek',
        #     'primals': True,
        #     'duals': True,
        #     'dualize': True,
        #     'abs_bnb_opt_tol': None,
        #     'abs_dual_fsb_tol': 1e-9,
        #     'abs_ipm_opt_tol': 1e-9,
        #     'abs_prim_fsb_tol': 1e-9,
        #     'integrality_tol': None,
        #     'markowitz_tol': None,
        #     'rel_bnb_opt_tol': None,
        #     'rel_dual_fsb_tol': 1e-9,
        #     'rel_ipm_opt_tol': 1e-9,
        #     'rel_prim_fsb_tol': 1e-9,
        # }

        problem = LmiEdmd._create_new_forw_base_problem(
            X_unshifted, X_shifted, self.alpha, self.spectral_radius,
            self.picos_eps, self.new_eps)

        problem.solve(**self.solver_params_, verbose=True)

        K = np.array(problem.get_valued_variable('K'), ndmin=2)
        B = np.array(problem.get_valued_variable('B'), ndmin=2)
        P = np.array(problem.get_valued_variable('P'),
                     ndmin=2) if self.P is None else self.P

        A = scipy.linalg.solve(P.T, K.T).T

        P_norm = np.linalg.norm(P, 'fro')
        diff_norm = np.linalg.norm(_G @ H_inv - np.block([[A, B]]), 'fro')

        min_val = {}
        min_val['{}'.format(self.new_eps)] = [P_norm, diff_norm]

        path = 'build/minimize_val/'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + '{}.bin'.format(self.new_eps), "wb") as f:
            data_dump = pickle.dump(min_val, f)

        coef = np.block([[A, B]])

        return coef.T

    def _validate_parameters(self) -> None:
        # Check spectral radius
        if self.spectral_radius <= 0:
            raise ValueError('`spectral_radius` must be positive.')
        if self.max_iter <= 0:
            raise ValueError('`max_iter` must be positive.')
        if self.iter_atol < 0:
            raise ValueError('`iter_atol` must be positive or zero.')
        if self.iter_rtol < 0:
            raise ValueError('`iter_rtol` must be positive or zero.')


class LmiEdmdSpectralRadiusConstrAvg(LmiRegressor):
    """LMI-based EDMD with spectral radius constraint for forward and backward. Inspired by the pykoop package. 
    """

    def __init__(self,
                 spectral_radius: float = 1.0,
                 new_eps: np.ndarray = 1e-2,
                 max_iter: int = 100,
                 iter_atol: float = 1e-6,
                 iter_rtol: float = 0,
                 alpha: float = 0,
                 inv_method: str = 'svd',
                 tsvd: tsvd.Tsvd = None,
                 picos_eps: float = 1e-5,
                 solver_params: Dict[str, Any] = None) -> None:
        """Instantiate :class:`LmiEdmdSpectralRadiusConstrAvg`. Inspired by the pykoop package.
        """
        self.spectral_radius = spectral_radius
        self.new_eps = new_eps
        self.max_iter = max_iter
        self.iter_atol = iter_atol
        self.iter_rtol = iter_rtol
        self.alpha = alpha
        self.inv_method = inv_method
        self.tsvd = tsvd
        self.picos_eps = picos_eps
        self.solver_params = solver_params

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        # Set solver parameters
        self.solver_params_ = self._default_solver_params.copy()
        if self.solver_params is not None:
            self.solver_params_.update(self.solver_params)
        # Clone TSVD
        self.tsvd_ = (sklearn.base.clone(self.tsvd)
                      if self.tsvd is not None else tsvd.Tsvd())
        # Get needed sizes
        p = X_unshifted.shape[1]
        p_theta = X_shifted.shape[1]
        # Make initial guesses and iterate
        P = np.eye(p_theta)
        # Set scope of other variables
        U = np.zeros((p_theta, p))
        self.objective_log_ = []

        # Avg
        var = np.var(X_shifted, axis=0)
        W = np.diag(1 / var)
        q = X_shifted.shape[0]
        p_theta = X_shifted.shape[1]
        Psi_f = X_unshifted.T
        Theta_plus_f = X_shifted.T
        n_inputs = Psi_f.shape[0] - Theta_plus_f.shape[0]
        Psi_b = np.zeros(Psi_f.shape)
        Psi_b[:-(n_inputs), :] = np.flip(Theta_plus_f, axis=1)
        Psi_b[-(n_inputs):, :-1] = np.flip(Psi_f[-(n_inputs):, :-1], axis=1)
        Theta_plus_b = np.flip(Psi_f[:-(n_inputs), :], axis=1)
        new_eps_og = np.linalg.norm(
            Psi_f.T @ scipy.linalg.pinv(Psi_f @ Psi_f.T), 2)
        # new_eps_og = 0.001

        _Gf = 1 / q * Theta_plus_f @ Psi_f.T
        _Hf = 1 / q * Psi_f @ Psi_f.T
        H_inv_f = scipy.linalg.pinv(_Hf)
        U_edmd_f = scipy.linalg.lstsq(_Hf.T, _Gf.T)[0].T
        _Gb = 1 / q * Theta_plus_b @ Psi_b.T
        _Hb = 1 / q * Psi_b @ Psi_b.T
        H_inv_b = scipy.linalg.pinv(_Hb)
        U_edmd_b = scipy.linalg.lstsq(_Hb.T, _Gb.T)[0].T

        if self.new_eps == -1:
            self.new_eps = new_eps_og

        # self.solver_params_ = {
        #     'solver': 'mosek',
        #     'primals': True,
        #     'duals': False,
        #     'dualize': False,
        #     'abs_bnb_opt_tol': None,
        #     'abs_dual_fsb_tol': 1e-9,
        #     'abs_ipm_opt_tol': 1e-9,
        #     'abs_prim_fsb_tol': 1e-9,
        #     'integrality_tol': None,
        #     'markowitz_tol': None,
        #     'rel_bnb_opt_tol': None,
        #     'rel_dual_fsb_tol': 1e-9,
        #     'rel_ipm_opt_tol': 1e-9,
        #     'rel_prim_fsb_tol': 1e-9,
        # }

        problem = LmiEdmd._create_new_avg_base_problem(X_unshifted, X_shifted,
                                                       self.alpha,
                                                       self.spectral_radius,
                                                       self.picos_eps,
                                                       self.new_eps)

        problem.solve(**self.solver_params_, verbose=True)

        P = np.array(problem.get_valued_variable('P'), ndmin=2)

        Kf = np.array(problem.get_valued_variable('Kf'), ndmin=2)
        Af = scipy.linalg.solve(P.T, Kf.T).T
        Bf = np.array(problem.get_valued_variable('Bf'), ndmin=2)

        Kb = np.array(problem.get_valued_variable('Kb'), ndmin=2)
        Bb = np.array(problem.get_valued_variable('Bb'), ndmin=2)
        Ab = scipy.linalg.solve(P.T, Kb.T).T

        P_tilde = np.block(
            [[P, np.zeros((P.shape[0], H_inv_f.shape[0] - P.shape[0]))],
             [
                 np.zeros((H_inv_f.shape[0] - P.shape[0], P.shape[0])),
                 np.eye(H_inv_f.shape[0] - P.shape[0])
             ]])

        A_squared = scipy.linalg.lstsq(Ab.T, Af.T)[0].T
        Afb = scipy.linalg.lstsq(Ab, np.eye(Ab.shape[0]))[0]
        Bfb = -Afb @ Bb
        temp = Bf + Af @ Bfb

        A = scipy.linalg.sqrtm(A_squared).real
        B = scipy.linalg.lstsq((np.eye(A.shape[0]) + A), temp)[0]

        matrix_dict = {}
        matrix_dict['Af'] = Af
        matrix_dict['Ab'] = Ab
        matrix_dict['A'] = A

        # New Epsilon experiment
        P_norm = np.linalg.norm(P, 'fro')
        diff_norm = np.linalg.norm(_Gf @ H_inv_f - np.block([[A, B]]), 'fro')

        min_val = {}
        min_val['{}'.format(self.new_eps)] = [P_norm, diff_norm]

        path = 'build/minimize_val/'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + '{}.bin'.format(self.new_eps), "wb") as f:
            data_dump = pickle.dump(min_val, f)

        return np.hstack([A, B]).T

    def _validate_parameters(self) -> None:
        # Check spectral radius
        if self.spectral_radius <= 0:
            raise ValueError('`spectral_radius` must be positive.')
        if self.max_iter <= 0:
            raise ValueError('`max_iter` must be positive.')
        if self.iter_atol < 0:
            raise ValueError('`iter_atol` must be positive or zero.')
        if self.iter_rtol < 0:
            raise ValueError('`iter_rtol` must be positive or zero.')


class LmiEdmdSpectralRadiusConstrAvgNoAS(LmiRegressor):
    """LMI-based EDMD with no spectral radius constraint for forward and backward. Inspired by the pykoop package.

   
    """

    def __init__(self,
                 spectral_radius: float = 1.0,
                 new_eps: np.ndarray = 1e-2,
                 max_iter: int = 100,
                 iter_atol: float = 1e-12,
                 iter_rtol: float = 0,
                 alpha: float = 0,
                 inv_method: str = 'svd',
                 tsvd: tsvd.Tsvd = None,
                 picos_eps: float = 1e-5,
                 solver_params: Dict[str, Any] = None) -> None:
        """Instantiate :class:`LmiEdmdSpectralRadiusConstrAvgNoAS'. Inspired by the pykoop package.

        """
        self.spectral_radius = spectral_radius
        self.new_eps = new_eps
        self.max_iter = max_iter
        self.iter_atol = iter_atol
        self.iter_rtol = iter_rtol
        self.alpha = alpha
        self.inv_method = inv_method
        self.tsvd = tsvd
        self.picos_eps = picos_eps
        self.solver_params = solver_params

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        # Set solver parameters
        self.solver_params_ = self._default_solver_params.copy()
        if self.solver_params is not None:
            self.solver_params_.update(self.solver_params)
        # Clone TSVD
        self.tsvd_ = (sklearn.base.clone(self.tsvd)
                      if self.tsvd is not None else tsvd.Tsvd())
        # Get needed sizes
        p = X_unshifted.shape[1]
        p_theta = X_shifted.shape[1]
        # Make initial guesses and iterate
        P = np.eye(p_theta)
        # Set scope of other variables
        U = np.zeros((p_theta, p))
        self.objective_log_ = []

        q = X_shifted.shape[0]
        p_theta = X_shifted.shape[1]
        Psi_f = X_unshifted.T
        Theta_plus_f = X_shifted.T
        n_inputs = Psi_f.shape[0] - Theta_plus_f.shape[0]
        Psi_b = np.zeros(Psi_f.shape)
        Psi_b[:-(n_inputs), :] = np.flip(Theta_plus_f, axis=1)
        Psi_b[-(n_inputs):, :-1] = np.flip(Psi_f[-(n_inputs):, :-1], axis=1)
        Theta_plus_b = np.flip(Psi_f[:-(n_inputs), :], axis=1)

        # self.solver_params_ = {
        #     'solver': 'mosek',
        #     'primals': True,
        #     'duals': True,
        #     'dualize': True,
        #     'abs_bnb_opt_tol': None,
        #     'abs_dual_fsb_tol': 1e-9,
        #     'abs_ipm_opt_tol': 1e-9,
        #     'abs_prim_fsb_tol': 1e-9,
        #     'integrality_tol': None,
        #     'markowitz_tol': None,
        #     'rel_bnb_opt_tol': None,
        #     'rel_dual_fsb_tol': 1e-9,
        #     'rel_ipm_opt_tol': 1e-9,
        #     'rel_prim_fsb_tol': 1e-9,
        # }

        kp_f = pykoop.Edmd(alpha=0)
        kp_f.fit(Psi_f.T, Theta_plus_f.T)
        Af = kp_f.coef_.T[:, :p_theta]
        Bf = kp_f.coef_.T[:, p_theta:]

        kp_b = pykoop.Edmd(alpha=0)
        kp_b.fit(Psi_b.T, Theta_plus_b.T)
        Ab = kp_b.coef_.T[:, :p_theta]
        Bb = kp_b.coef_.T[:, p_theta:]

        Afb = scipy.linalg.lstsq(Ab, np.eye(Ab.shape[0]))[0]
        Bfb = -Afb @ Bb
        A_squared = scipy.linalg.lstsq(Ab.T, Af.T)[0].T
        temp = Bf + Af @ Bfb

        A = scipy.linalg.sqrtm(A_squared).real
        B = scipy.linalg.lstsq((np.eye(A.shape[0]) + A), temp)[0]

        return np.hstack([A, B]).T

    def _validate_parameters(self) -> None:
        # Check spectral radius
        if self.spectral_radius <= 0:
            raise ValueError('`spectral_radius` must be positive.')
        if self.max_iter <= 0:
            raise ValueError('`max_iter` must be positive.')
        if self.iter_atol < 0:
            raise ValueError('`iter_atol` must be positive or zero.')
        if self.iter_rtol < 0:
            raise ValueError('`iter_rtol` must be positive or zero.')


class Tedmd(LmiRegressor):
    """LMI-based EDMD with no spectral radius constraint for forward and backward. Inspired by the pykoop package.

   
    """

    def __init__(self,
                 spectral_radius: float = 0.99999,
                 new_eps: np.ndarray = -1,
                 max_iter: int = 100,
                 iter_atol: float = 1e-12,
                 iter_rtol: float = 0,
                 alpha: float = 0,
                 inv_method: str = 'svd',
                 tsvd: tsvd.Tsvd = None,
                 picos_eps: float = 1e-5,
                 solver_params: Dict[str, Any] = None) -> None:
        """Instantiate :class:`LmiEdmdSpectralRadiusConstrAvgNoAS'. Inspired by the pykoop package.

        """
        self.spectral_radius = spectral_radius
        self.new_eps = new_eps
        self.max_iter = max_iter
        self.iter_atol = iter_atol
        self.iter_rtol = iter_rtol
        self.alpha = alpha
        self.inv_method = inv_method
        self.tsvd = tsvd
        self.picos_eps = picos_eps
        self.solver_params = solver_params

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        # Set solver parameters
        self.solver_params_ = self._default_solver_params.copy()
        if self.solver_params is not None:
            self.solver_params_.update(self.solver_params)
        # Clone TSVD
        self.tsvd_ = (sklearn.base.clone(self.tsvd)
                      if self.tsvd is not None else tsvd.Tsvd())
        # Get needed sizes
        p = X_unshifted.shape[1]
        p_theta = X_shifted.shape[1]
        # Make initial guesses and iterate
        P = np.eye(p_theta)
        # Set scope of other variables
        U = np.zeros((p_theta, p))
        self.objective_log_ = []

        q = X_shifted.shape[0]
        p_theta = X_shifted.shape[1]
        Psi = X_unshifted.T
        Theta_plus = X_shifted.T
        n_inputs = Psi.shape[0] - Theta_plus.shape[0]

        # Check the variance
        with open("build/others/variance.bin", "rb") as f:
            variance = pickle.load(f)

        with open("build/others/robot.bin", "rb") as f:
            robot = pickle.load(f)

        # Debiasing process
        Z = np.concatenate((Psi[:, :], Theta_plus[:, :]), axis=0)
        U, S, V = np.linalg.svd(Z, full_matrices=False)
        b = Z.shape[0] / Z.shape[1]
        w = 0.56 * b**3 - 0.95 * b**2 + 1.82 * b + 1.43
        S_threshold = w * statistics.median(S)
        # Obtain the number of singular values above S_threshold
        # n_sv = np.sum(S > S_threshold)
        n_sv = sv_finder(variance, robot)
        Q = V[:n_sv, :].T

        X_hat = Psi[:, :] @ Q
        Y_hat = Theta_plus[:, :] @ Q

        _G = 1 / q * Y_hat @ X_hat.T
        _H = 1 / q * X_hat @ X_hat.T
        H_inv = scipy.linalg.pinv(_H)

        U_hat = scipy.linalg.lstsq(X_hat.T, Y_hat.T)[0].T
        A = U_hat[:p_theta, :]

        kp = pykoop.Edmd(alpha=0)
        kp.fit((Psi @ Q).T, (Theta_plus @ Q).T)
        A = kp.coef_.T[:, :p_theta]
        B = kp.coef_.T[:, p_theta:]

        return np.hstack([A, B]).T

    def _validate_parameters(self) -> None:
        # Check spectral radius
        if self.spectral_radius <= 0:
            raise ValueError('`spectral_radius` must be positive.')
        if self.max_iter <= 0:
            raise ValueError('`max_iter` must be positive.')
        if self.iter_atol < 0:
            raise ValueError('`iter_atol` must be positive or zero.')
        if self.iter_rtol < 0:
            raise ValueError('`iter_rtol` must be positive or zero.')


class TedmdAS(LmiRegressor):
    """LMI-based EDMD with no spectral radius constraint for forward and backward. Inspired by the pykoop package.

   
    """

    def __init__(self,
                 spectral_radius: float = 0.99999,
                 new_eps: np.ndarray = -1,
                 max_iter: int = 100,
                 iter_atol: float = 1e-12,
                 iter_rtol: float = 0,
                 alpha: float = 0,
                 inv_method: str = 'svd',
                 tsvd: tsvd.Tsvd = None,
                 picos_eps: float = 1e-5,
                 solver_params: Dict[str, Any] = None) -> None:
        """Instantiate :class:`LmiEdmdSpectralRadiusConstrAvgNoAS'. Inspired by the pykoop package.

        """
        self.spectral_radius = spectral_radius
        self.new_eps = new_eps
        self.max_iter = max_iter
        self.iter_atol = iter_atol
        self.iter_rtol = iter_rtol
        self.alpha = alpha
        self.inv_method = inv_method
        self.tsvd = tsvd
        self.picos_eps = picos_eps
        self.solver_params = solver_params

    def _fit_regressor(self, X_unshifted: np.ndarray,
                       X_shifted: np.ndarray) -> np.ndarray:
        # Set solver parameters
        self.solver_params_ = self._default_solver_params.copy()
        if self.solver_params is not None:
            self.solver_params_.update(self.solver_params)
        # Clone TSVD
        self.tsvd_ = (sklearn.base.clone(self.tsvd)
                      if self.tsvd is not None else tsvd.Tsvd())
        # Get needed sizes
        p = X_unshifted.shape[1]
        p_theta = X_shifted.shape[1]
        # Make initial guesses and iterate
        P = np.eye(p_theta)
        # Set scope of other variables
        U = np.zeros((p_theta, p))
        self.objective_log_ = []

        q = X_shifted.shape[0]
        p_theta = X_shifted.shape[1]
        Psi = X_unshifted.T
        Theta_plus = X_shifted.T
        n_inputs = Psi.shape[0] - Theta_plus.shape[0]

        # Check how many data points are in each episode
        eps = np.zeros((1, Psi.shape[1]))
        # Upload episode feature
        with open("build/others/ep_feat.bin", "rb") as f:
            data = pickle.load(f)

        for i in range(int(np.max(data)) + 1):
            idx = data[0, data[0, :] == i][:-1]
            eps = idx if i == 0 else np.concatenate((eps, idx), axis=0)

        eps_0 = eps[eps == 0].shape[0]

        # Check the variance
        with open("build/others/variance.bin", "rb") as f:
            variance = pickle.load(f)

        with open("build/others/robot.bin", "rb") as f:
            robot = pickle.load(f)

        # Debiasing process
        Z = np.concatenate((Psi[:, :], Theta_plus[:, :]), axis=0)
        U, s, V = np.linalg.svd(Z, full_matrices=False)
        b = Z.shape[0] / Z.shape[1]
        w = 0.56 * b**3 - 0.95 * b**2 + 1.82 * b + 1.43
        s_thresh = w * statistics.median(s)
        b = np.array([[b]])

        # n_sv = sv_finder(variance, robot)

        # Q = V[:n_sv, :].T

        # X_hat = Psi[:, :] @ Q
        # Y_hat = Theta_plus[:, :] @ Q

        # _G = 1 / q * Y_hat @ X_hat.T
        # _H = 1 / q * X_hat @ X_hat.T
        # H_inv = scipy.linalg.pinv(_H)

        # U_hat = scipy.linalg.lstsq(X_hat.T, Y_hat.T)[0].T
        # A = U_hat[:p_theta, :]

        # new_eps_og = np.linalg.norm(
        #     X_hat.T @ scipy.linalg.pinv(X_hat @ X_hat.T), 2)

        # if self.new_eps == -1:
        #     self.new_eps = new_eps_og

        # problem = LmiEdmd._create_tedmd_AS_problem(
        #     (Psi[:, :] @ Q).T, (Theta_plus @ Q).T, self.alpha,
        #     self.spectral_radius, self.picos_eps, self.new_eps)

        # problem.solve(**self.solver_params_, verbose=True)
        # K = np.array(problem.get_valued_variable('K'), ndmin=2)
        # P = np.array(problem.get_valued_variable('P'), ndmin=2)
        # A = scipy.linalg.solve(P.T, K.T).T
        # B = np.array(problem.get_valued_variable('B'), ndmin=2)

        # path = "build/others/"
        # os.makedirs(os.path.dirname(path), exist_ok=True)
        # with open(path + "n_sv_to_keep.bin".format(n_sv), "wb") as f:
        #     data_dump = pickle.dump(n_sv, f)

        # return np.hstack([A, B]).T

        if variance == 0:
            try:
                with open("build/others/n_sv_to_keep.bin", "rb") as f:
                    n_sv = pickle.load(f)
            except:
                n_sv = sv_finder(variance, robot)

            Q = V[:n_sv, :].T
            X_hat = Psi[:, :] @ Q
            Y_hat = Theta_plus[:, :] @ Q
            new_eps_og = np.linalg.norm(
                X_hat.T @ scipy.linalg.pinv(X_hat @ X_hat.T), 2)
            if self.new_eps == -1:
                self.new_eps = new_eps_og
            problem = LmiEdmd._create_tedmd_AS_problem(
                (Psi[:, :] @ Q).T, (Theta_plus @ Q).T, self.alpha,
                self.spectral_radius, self.picos_eps, self.new_eps)

            problem.solve(**self.solver_params_, verbose=True)
            K = np.array(problem.get_valued_variable('K'), ndmin=2)
            P = np.array(problem.get_valued_variable('P'), ndmin=2)
            A = scipy.linalg.solve(P.T, K.T).T
            B = np.array(problem.get_valued_variable('B'), ndmin=2)

        else:
            n_sv = sv_finder(variance, robot)

            Q = V[:n_sv, :].T

            X_hat = Psi[:, :] @ Q
            Y_hat = Theta_plus[:, :] @ Q

            _G = 1 / q * Y_hat @ X_hat.T
            _H = 1 / q * X_hat @ X_hat.T
            H_inv = scipy.linalg.pinv(_H)

            U_hat = scipy.linalg.lstsq(X_hat.T, Y_hat.T)[0].T
            A = U_hat[:p_theta, :]

            new_eps_og = np.linalg.norm(
                X_hat.T @ scipy.linalg.pinv(X_hat @ X_hat.T), 2)
            # new_eps_og = 0.001

            if self.new_eps == -1:
                self.new_eps = new_eps_og

            problem = LmiEdmd._create_tedmd_AS_problem(
                (Psi[:, :] @ Q).T, (Theta_plus @ Q).T, self.alpha,
                self.spectral_radius, self.picos_eps, self.new_eps)

            problem.solve(**self.solver_params_, verbose=True)
            K = np.array(problem.get_valued_variable('K'), ndmin=2)
            P = np.array(problem.get_valued_variable('P'), ndmin=2)
            A = scipy.linalg.solve(P.T, K.T).T
            B = np.array(problem.get_valued_variable('B'), ndmin=2)

        path = "build/others/"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + "n_sv_to_keep.bin".format(n_sv), "wb") as f:
            data_dump = pickle.dump(n_sv, f)

        return np.hstack([A, B]).T

    def _validate_parameters(self) -> None:
        # Check spectral radius
        if self.spectral_radius <= 0:
            raise ValueError('`spectral_radius` must be positive.')
        if self.max_iter <= 0:
            raise ValueError('`max_iter` must be positive.')
        if self.iter_atol < 0:
            raise ValueError('`iter_atol` must be positive or zero.')
        if self.iter_rtol < 0:
            raise ValueError('`iter_rtol` must be positive or zero.')
