# ================================================================================
# Created By : Dominic Paragas
# Date Created : Sat September 10 23:00:00 PDT 2022
# Date Last Modified: Thu September 22 2022
# Python Version: 3.10
# Email: dparagas@berkeley.edu
# ================================================================================
"""This collection of functions was made for UC Berkeley's Physics 5CL course."""
# ================================================================================
# Imports
# ================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import ndarray
from pandas.core.frame import DataFrame

# make arrays fast

def make_array(*args):
    return np.array(args)

def ma(*args):
    return np.array(args)

def qarr(*args):
    return np.array(args)

def to_array(iterable) -> ndarray:
    return np.array(list(iterable))

# statistics

def mean(iterable) -> float:
    """Returns the mean of an iterable's elements"""
    arr = to_array(iterable).flatten()
    result = np.sum(arr) / len(arr)
    return float(result)


def dev_mean(iterable) -> ndarray:
    """Returns the deviation from the mean for each element of an iterable."""
    arr = to_array(iterable)
    return arr - mean(arr)


def var_parent(iterable) -> float:
    """Returns the variance of an iterable's elements,
    representing a parent distribution."""
    arr = to_array(iterable).flatten()
    result = mean(dev_mean(arr) ** 2)
    return float(result)


def var_sample(iterable) -> float:
    """Returns the variance of an iterable's elements,
    representing a sample distribution."""
    arr = to_array(iterable).flatten()
    result = np.sum(dev_mean(arr) ** 2) / (len(arr) - 1)
    return float(result)


def std_parent(iterable) -> float:
    """Returns the standard deviation of an iterable's elements,
    representing  a parent distribution."""
    arr = to_array(iterable).flatten()
    n = len(arr)
    inside = np.sum((arr - mean(arr)) ** 2) / n
    return float(np.sqrt(inside))


def std_sample(iterable) -> float:
    """Returns the standard deviation of an iterable's elements,
    representing  a sample distribution."""
    arr = to_array(iterable).flatten()
    n = len(arr)
    inside = np.sum((arr - mean(arr)) ** 2) / (n - 1)
    return float(np.sqrt(inside))


def std(iterable, type="parent") -> float:
    """Returns the standard deviation of an iterable's elements.
    The `type` parameter specifies if the iterable represents a
    parent or sample distribution. Also, if the iterable has less
    than five elements, the sample standard deviation is taken. """
    assert type == 'parent' or type == 'sample'
    arr = to_array(iterable).flatten()
    if (len(arr) >= 5) and (type == "parent"):
        return std_parent(arr)
    else:
        return std_sample(arr)


def standard_err(iterable) -> float:
    """Returns the standard error of an iterable's elements."""
    arr = to_array(iterable).flatten()
    n = len(arr)
    sigma = std_sample(arr)
    return float(sigma / np.sqrt(n))


assertion_xy_lens = "the x-values array and y-values array " \
                    "need to have the same length"


def covariance(x_iterable, y_iterable):
    """Returns the covariance of the elements of two iterables."""
    x_arr = to_array(x_iterable).flatten()
    y_arr = to_array(y_iterable).flatten()
    assert len(x_arr) == len(y_arr), assertion_xy_lens
    n = len(x_arr)
    summ_x = x_arr - np.array([mean(x_arr)] * np.size(x_arr))
    summ_y = y_arr - np.array([mean(y_arr)] * np.size(y_arr))
    summation = np.sum(summ_x * summ_y)
    result = (1 / (n - 1)) * summation
    return float(result)


def corr_coeff(x_iterable, y_iterable) -> float:
    """Returns the correlation coefficient of the elements of two iterables."""
    x_arr = to_array(x_iterable).flatten()
    y_arr = to_array(y_iterable).flatten()
    cov = covariance(x_arr, y_arr)
    sigma_x = std_parent(x_arr)
    sigma_y = std_parent(y_arr)
    result = cov / (sigma_x * sigma_y)
    return float(result)


def mean_pm_ste():
    # TODO
    return False


def prop_uncty_singvar():
    # TODO
    return False


def prop_uncty_multvar():
    # TODO
    return False

def resid(x_iterable, y_iterable, uncty_iterable, reg_model: type) -> ndarray:
    """Returns the residuals using iterables."""
    x_arr = to_array(x_iterable).flatten()
    y_arr = to_array(y_iterable).flatten()
    uncty_arr = to_array(uncty_iterable).flatten()
    return y_arr - reg_model.model(x_arr, y_arr, uncty_arr)


def norm_resid(x_iterable, y_iterable, uncty_iterable, reg_model: type) -> ndarray:
    """Returns the normalized residuals using iterables."""
    x_arr = to_array(x_iterable).flatten()
    y_arr = to_array(y_iterable).flatten()
    uncty_arr = to_array(uncty_iterable).flatten()
    return resid(x_arr, y_arr, uncty_arr, reg_model) / uncty_arr


def chi_sq(x_iterable, y_iterable, uncty_iterable, reg_model: type) -> float:
    """Returns the chi squared using iterables."""
    x_arr = to_array(x_iterable).flatten()
    y_arr = to_array(y_iterable).flatten()
    uncty_arr = to_array(uncty_iterable).flatten()
    return np.sum(norm_resid(x_arr, y_arr, uncty_arr, reg_model) ** 2)


class SimpleLin:
    name = "Simple Linear Model"
    equation = "y=mx+c"
    param = 2

    def model(x_arr: ndarray, y_arr: ndarray, uncty_arr=np.empty(0)) -> ndarray:
        assert len(x_arr) == len(y_arr), assertion_xy_lens
        m: float = SimpleLin.slope(x_arr, y_arr)
        c: float = SimpleLin.intercept(x_arr, y_arr)
        x: ndarray = x_arr
        y: ndarray = m * x + c
        return y

    def delta_combo(x_arr: ndarray) -> float:
        n: int = len(x_arr)
        ins_sum_sq = np.sum(x_arr ** 2)
        out_sum_sq = (np.sum(x_arr)) ** 2
        delta = n * ins_sum_sq - out_sum_sq
        return float(delta)

    def slope(x_arr: ndarray, y_arr: ndarray, uncty_arr=np.empty(0)) -> float:
        assert len(x_arr) == len(y_arr), assertion_xy_lens
        cov_xy = covariance(x_arr, y_arr)
        sigma_x = std(x_arr)
        m = cov_xy / (sigma_x ** 2)
        return float(m)

    def intercept(x_arr: ndarray, y_arr: ndarray, uncty_arr=np.empty(0)) -> float:
        assert len(x_arr) == len(y_arr), assertion_xy_lens
        y_bar = mean(y_arr)
        x_bar = mean(x_arr)
        m: float = SimpleLin.slope(x_arr, y_arr)
        c: float = y_bar - m * x_bar
        return c

    def common_uncty(x_arr: ndarray, y_arr: ndarray) -> float:
        assert len(x_arr) == len(y_arr), assertion_xy_lens
        n = len(x_arr)
        m: float = SimpleLin.slope(x_arr, y_arr)
        c: float = SimpleLin.intercept(x_arr, y_arr)
        ins_summ: ndarray = (y_arr - m * x_arr - c) ** 2
        return np.sqrt(np.sum(ins_summ) / (n - 2))

    def slope_uncty(x_arr: ndarray, y_arr: ndarray, uncty_arr=np.empty(0)) -> float:
        assert len(x_arr) == len(y_arr), assertion_xy_lens
        n = len(x_arr)
        alpha_cu = SimpleLin.common_uncty(x_arr, y_arr)
        sigma_x = std(x_arr)
        alpha_m = alpha_cu / (np.sqrt(n * (sigma_x ** 2)))
        return float(alpha_m)

    def intercept_uncty(x_arr: ndarray, y_arr: ndarray, uncty_arr=np.empty(0)) -> float:
        assert len(x_arr) == len(y_arr), assertion_xy_lens
        alpha_m = SimpleLin.slope_uncty(x_arr, y_arr)
        x_sq_bar: float = mean(x_arr ** 2)
        return alpha_m * np.sqrt(x_sq_bar)


class SimpleDirectProp:
    name = "Simple Direct Proportionality Model"
    equation = "y=mx"
    param = 1

    def model(x_arr: ndarray, y_arr: ndarray, uncty_arr=np.empty(0)) -> ndarray:
        assert len(x_arr) == len(y_arr), assertion_xy_lens
        m = SimpleDirectProp.slope(x_arr, y_arr)
        x = x_arr
        y: ndarray = m * x
        return y

    def slope(x_arr: ndarray, y_arr: ndarray, uncty_arr=np.empty(0)) -> float:
        assert len(x_arr) == len(y_arr), assertion_xy_lens
        xy_bar: float = mean(x_arr * y_arr)
        x_sq_bar: float = mean(x_arr ** 2)
        return xy_bar / x_sq_bar

    def common_uncty(x_arr: ndarray, y_arr: ndarray) -> float:
        assert len(x_arr) == len(y_arr), assertion_xy_lens
        n = len(x_arr)
        m = SimpleDirectProp.slope(x_arr, y_arr)
        ins_summ = (y_arr - m * x_arr) ** 2
        return np.sqrt(np.sum(ins_summ) / (n - 1))

    def slope_uncty(x_arr: ndarray, y_arr: ndarray, uncty_arr=np.empty(0)) -> float:
        assert len(x_arr) == len(y_arr), assertion_xy_lens
        n = len(x_arr)
        alpha_cu = SimpleDirectProp.common_uncty(x_arr, y_arr)
        x_sq_bar = mean(x_arr ** 2)
        alpha_m = alpha_cu / (np.sqrt(n * x_sq_bar))
        return float(alpha_m)


def weights(uncty_arr):
    return 1 / (uncty_arr ** 2)


assertion_xyu_lens = "the x-values array &/or y-values array and " \
                     "uncertainty-values array need to have the same length"


class WeightLin:
    name = "Weighted Linear Model"
    equation = "y=mx+c"
    param = 2

    def model(x_arr: ndarray, y_arr: ndarray, uncty_arr: ndarray) -> ndarray:
        assert len(x_arr) == len(y_arr), assertion_xy_lens
        assert len(x_arr) == len(uncty_arr), assertion_xyu_lens
        m: float = WeightLin.slope(x_arr, y_arr, uncty_arr)
        c: float = WeightLin.intercept(x_arr, y_arr, uncty_arr)
        x: ndarray = x_arr
        y: ndarray = m * x + c
        return y

    def delta_combo(x_arr: ndarray, uncty_arr: ndarray) -> float:
        assert len(x_arr) == len(uncty_arr), assertion_xyu_lens
        w_arr = weights(uncty_arr)
        delta = np.sum(w_arr) * np.sum(w_arr * x_arr ** 2) - \
                (np.sum(w_arr * x_arr)) ** 2
        return float(delta)

    def slope(x_arr: ndarray, y_arr: ndarray, uncty_arr: ndarray) -> float:
        assert len(x_arr) == len(y_arr), assertion_xy_lens
        assert len(x_arr) == len(uncty_arr), assertion_xyu_lens
        w_arr = weights(uncty_arr)
        delta: float = WeightLin.delta_combo(x_arr, uncty_arr)
        numer = np.sum(w_arr) * np.sum(w_arr * x_arr * y_arr) - \
                np.sum(w_arr * x_arr) * np.sum(w_arr * y_arr)
        return numer / delta

    def intercept(x_arr: ndarray, y_arr: ndarray, uncty_arr: ndarray) -> float:
        assert len(x_arr) == len(y_arr), assertion_xy_lens
        assert len(x_arr) == len(uncty_arr), assertion_xyu_lens
        w_arr = weights(uncty_arr)
        m: float = WeightLin.slope(x_arr, y_arr, uncty_arr)
        numer = np.sum(w_arr * y_arr) - m * np.sum(w_arr * x_arr)
        return numer / np.sum(w_arr)

    def slope_uncty(x_arr: ndarray, y_arr: ndarray, uncty_arr: ndarray) -> float:
        w_arr = weights(uncty_arr)
        delta: float = WeightLin.delta_combo(x_arr, uncty_arr)
        return np.sqrt(np.sum(w_arr) / delta)

    def intercept_uncty(x_arr: ndarray, y_arr: ndarray, uncty_arr: ndarray) -> float:
        w_arr = weights(uncty_arr)
        delta: float = WeightLin.delta_combo(x_arr, uncty_arr)
        return np.sqrt(np.sum(w_arr * x_arr ** 2) / delta)


class WeightDirectProp:
    name = "Weighted Direct Proportionality Model"
    equation = "y=mx"
    param = 1

    def model(x_arr: ndarray, y_arr: ndarray, uncty_arr: ndarray) -> ndarray:
        assert len(x_arr) == len(y_arr), assertion_xy_lens
        assert len(x_arr) == len(uncty_arr), assertion_xyu_lens
        m: float = WeightDirectProp.slope(x_arr, y_arr, uncty_arr)
        x: ndarray = x_arr
        y: ndarray = m * x
        return y

    def slope(x_arr: ndarray, y_arr: ndarray, uncty_arr: ndarray) -> float:
        assert len(x_arr) == len(y_arr), assertion_xy_lens
        assert len(x_arr) == len(uncty_arr), assertion_xyu_lens
        w_arr = weights(uncty_arr)
        numer = np.sum(w_arr * x_arr * y_arr)
        denom = np.sum(w_arr * x_arr ** 2)
        m = numer / denom
        return float(m)

    def slope_uncty(x_arr: ndarray, y_arr: ndarray, uncty_arr: ndarray) -> float:
        assert len(x_arr) == len(y_arr), assertion_xy_lens
        assert len(x_arr) == len(uncty_arr), assertion_xyu_lens
        n = len(x_arr)
        m: float = WeightDirectProp.slope(x_arr, y_arr, uncty_arr)
        numer_r = np.sum((y_arr - m * x_arr) ** 2)
        denom_r = np.sum(x_arr ** 2)
        left = 1 / (n - 1)
        return np.sqrt(left * (numer_r / denom_r))


def uncty_indp_equiv(indp_arr: ndarray, depd_arr: ndarray, indp_uncty: ndarray, depd_uncty: ndarray) -> ndarray:
    m_simple = SimpleLin.slope(indp_arr, depd_arr)
    return np.sqrt(depd_uncty ** 2 + ((m_simple ** 2) * (indp_uncty ** 2)))


def agreement_test(y, z, uncty_y, uncty_z):
    """Returns true if we claim agreement and false otherwise."""
    numer = np.abs(y - z)
    denom = 2 * np.sqrt(uncty_y ** 2 + uncty_z ** 2)
    agr_test = numer / denom
    if agr_test < 1:
        return True
    else:
        return False


def coeff_deter():
    # TODO
    return False


def adj_coeff_deter():
    # TODO
    return False


def red_chi_sq(x_iterable, y_iterable, uncty_iterable, reg_model: type) -> float:
    """Returns the reduced chi squared using iterables."""
    x_arr = to_array(x_iterable).flatten()
    y_arr = to_array(y_iterable).flatten()
    uncty_arr = to_array(uncty_iterable).flatten()
    dof: float = deg_of_freedom(x_arr, y_arr, reg_model)
    chisq = chi_sq(x_arr, y_arr, uncty_arr, reg_model)
    return chisq / dof


def deg_of_freedom(x_iterable, y_iterable, reg_model: type) -> float:
    """Returns the degrees of freedom using iterables."""
    x_arr = to_array(x_iterable).flatten()
    y_arr = to_array(y_iterable).flatten()
    assert len(x_arr) == len(y_arr), assertion_xy_lens
    n = len(x_arr)
    params = reg_model.param
    nu = n - params
    return float(nu)

# Conversion functions

nm_to_m = lambda nm: nm * 1e-9
nm_to_cm = lambda nm: nm * 1e-7
m_to_cm = lambda m: m*100
cm_to_mm = lambda cm: cm*10
px_to_micro_m = lambda px: np.array((px*2.54, px*0.03))
micro_m_to_cm = lambda micro_m: micro_m*1e-4
micro_m_to_mm = lambda micro_m: micro_m*0.001
px_to_cm = lambda px: micro_m_to_cm(px_to_micro_m(px))
px_to_mm = lambda px: micro_m_to_mm(px_to_micro_m(px))

class error_analysis:

    def add(*terms):
        return np.sum(terms)

    def multiply(computed_val_s__without_uncty, terms, uncties):
        if terms != np.ndarray:
            terms = np.array([terms])
        if uncties != np.ndarray:
            uncties = np.array([uncties])
        assert len(uncties) == len(terms), "there is an unequal number of terms and uncertainties"
        frac_add = np.sum([uncties[i]/np.abs(terms[i]) for i in range(len(terms))])
        return frac_add*np.abs(computed_val_s__without_uncty)


def scatter_plt_arr(x_arr: ndarray, y_arr: ndarray, uncty_arr=None, ind_uncty_arr=None,
                    x_title="", y_title="", x_units="", y_units="", title="",
                    x_font_size=13, y_font_size=13, title_font_size=14, x_tick_size=10, y_tick_size=10,
                    figsize=(6, 6), ax=None):
    # Set figure axes
    if ax == None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.grid()


    plt.rc('xtick', labelsize=x_tick_size) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=y_tick_size) #fontsize of the y tick labels

    # Set title
    if (x_title != "") and (y_title != "") and (title == ""):
        title = x_title + " vs. " + y_title
    x_title, x_units = separate_title_units(x_title, x_units, "X-axis")
    x_label = x_title
    if x_units != "":
        x_label = x_title + f" ({x_units})"
    ax.set_xlabel(x_label, fontsize = x_font_size) # X label
    y_title, y_units = separate_title_units(y_title, y_units, "Y-axis")
    y_label = y_title
    if y_units != "":
        y_label = y_title + f" ({y_units})"
    ax.set_ylabel(y_label, fontsize = y_font_size) # Y label
    ax.set_title(title, fontweight="bold", size=title_font_size) # Title

    # Create scatter plot
    if uncty_arr is None:
        plt.scatter(x_arr, y_arr, label="Data without error bar")

    # Create error bars
    if (uncty_arr is not None) or (ind_uncty_arr is not None):
        plt.errorbar(x_arr, y_arr, yerr=uncty_arr, xerr=ind_uncty_arr, fmt='ko', capsize=3, capthick=1, label="data with error bar")

    if ax==None:
        # Create legend
        plt.legend(loc='upper left')
        plt.show()


def scatter_plt_df(df: DataFrame, x_title: str, y_title: str, uncty_title=None, ind_uncty_title=None,
                   x_units="", y_units="", title="",
                   figsize=(6, 6), x_font_size=13, y_font_size=13, title_font_size=14, x_tick_size=10, y_tick_size=10,
                   ax=None):
    x_vals = df[x_title].values
    y_vals = df[y_title].values
    if uncty_title is not None:
        uncty_vals = df[uncty_title].values
    if ind_uncty_title is not None:
        ind_uncty_vals = df[ind_uncty_title].values
    return scatter_plt_arr(x_vals, y_vals, uncty_arr=uncty_vals, ind_uncty_arr=ind_uncty_vals,
                           x_title=x_title, y_title=y_title, x_units=x_units, y_units=y_units, title=title,
                           figsize=figsize, x_font_size=x_font_size, y_font_size=y_font_size,
                           title_font_size=title_font_size, x_tick_size=x_tick_size, y_tick_size=y_tick_size, ax=ax)


def resid_plt_arr(x_arr, y_arr, uncty_arr, reg_model: type, x_title="", x_units="", resid_units="",
                  figsize=(6, 6), x_font_size=12, y_font_size=12,
                  title_font_size=14, x_tick_size=10, y_tick_size=10, grid=False):
    # Residuals
    resids = resid(x_arr, y_arr, uncty_arr, reg_model)

    # Sets figure size etc.
    fig, ax = plt.subplots(figsize=figsize)
    if grid:
        ax.grid()
    #ax.legend(loc="best")
    plt.rc('xtick', labelsize=x_tick_size) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=y_tick_size) #fontsize of the y tick labels

    # Residuals
    plt.errorbar(x_arr, resids, fmt='o')
    plt.axhline(color='r')  # 0 line for reference

    # Axes label and title
    fit_type = reg_model.name.replace("Model", "Fit")
    ax.set_title(fit_type + " Residuals", fontweight="bold", size=title_font_size) # Title
    x_title, x_units = separate_title_units(x_title, x_units, "X-axis")
    x_label = x_title
    if x_units != "":
        x_label = x_title + f" ({x_units})"
    ax.set_xlabel(x_label, fontsize = x_font_size) # X label
    resid_label = "Residuals: y(observed) - y (predicted)"
    if resid_units == "":
        print("Residual units not found")
    if resid_units != "":
        resid_label = f"Residuals: y(observed) - y(predicted)  ({resid_units})"
    ax.set_ylabel(resid_label, fontsize = y_font_size) # Y label
    plt.show()


def resid_plt_df(df: DataFrame, x_title: str, y_title: str, uncty_title: str, reg_model: type,
                 x_units="", resid_units="",
                 figsize=(6, 6), x_font_size=12, y_font_size=12,
                 title_font_size=14, x_tick_size=10, y_tick_size=10, grid=False):
    x_vals = df[x_title].values
    y_vals = df[y_title].values
    uncty_vals = df[uncty_title].values
    return resid_plt_arr(x_vals, y_vals, uncty_vals, reg_model, x_title=x_title, x_units=x_units,
                         resid_units=resid_units, figsize=figsize, x_font_size=x_font_size,
                         y_font_size=y_font_size, title_font_size=title_font_size,
                         x_tick_size=x_tick_size, y_tick_size=y_tick_size, grid=grid)


def separate_title_units(title: str, units: str, axis: str):
    if title == "":
        print(f"{axis} title not found")
        return title, units
    elif units == "":
        if not has_units(title):
            print(f"{title} units not found")
            return title, units
        else:
            title, units = split_title_units(title)
            return title, units
    else:
        if not has_units(title):
            return title, units
        else:
            title = split_title_units(title)[0]
            return title, units


def split_title_units(title: str):
    index_l = title.rindex('(')
    index_r = title.rindex(')')
    units = title[index_l + 1:index_r]
    title = title[:index_l]
    while title[-1] == " ":
        title = title[:-1]
    return title, units


def has_units(title: str):
    if ("(" not in title) or (")" not in title):
        return False
    index_l = title.rindex('(')
    index_r = title.rindex(')')
    units = title[index_l + 1:index_r]
    return True if units != "" else False


def best_fit_plt_arr(x_arr, y_arr, uncty_arr, reg_model: type,
                     x_title="", y_title="", x_units="", y_units="", title="",
                     figsize=(6, 6), x_font_size=12, y_font_size=12,
                     title_font_size=14, x_tick_size=10, y_tick_size=10, grid=False, show_ref=False, raw_data=False):

    # Set figure size and axes
    fig, ax = plt.subplots(figsize=figsize)
    if grid:
        ax.grid()

    range_min = find_plot_range_extrema(y_arr, uncty_arr, "min")
    range_max = find_plot_range_extrema(y_arr, uncty_arr, "max")

    xmarg = (max(x_arr) - min(x_arr)) * plt.margins()[0]
    ymarg = (range_max - range_min) * plt.margins()[1]

    plot_domain_min = min(x_arr) - xmarg
    plot_domain_max = max(x_arr) + xmarg
    plot_range_min = range_min - ymarg
    plot_range_max = range_max + ymarg

    plt.xlim(plot_domain_min, plot_domain_max)
    plt.ylim(plot_range_min, plot_range_max)
    plt.rc('xtick', labelsize=x_tick_size) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=y_tick_size) #fontsize of the y tick labels

    # Set title and axes labels
    fit_type = reg_model.name.replace("Model", "Fit")
    if (x_title != "") and (y_title != "") and (title == ""):
        title = x_title + " vs. " + y_title + "; " + fit_type
    x_title, x_units = separate_title_units(x_title, x_units, "X-axis")
    x_label = x_title
    if x_units != "":
        x_label = x_title + f" ({x_units})"
    ax.set_xlabel(x_label, fontsize = x_font_size) # X label
    y_title, y_units = separate_title_units(y_title, y_units, "Y-axis")
    y_label = y_title
    if y_units != "":
        y_label = y_title + f" ({y_units})"
    ax.set_ylabel(y_label, fontsize = y_font_size) # Y label
    ax.set_title(title, fontweight="bold", size=title_font_size) # Title

    # Plot data with error bars
    if uncty_arr is None:
        plt.errorbar(x_arr, y_arr, fmt='o', capsize = 3, capthick = 1, label="Data without error bar")
    if uncty_arr is not None:
        plt.errorbar(x_arr, y_arr, yerr=uncty_arr, fmt='o', capsize = 3, capthick = 1, label="Data with error bar")

    # Plot fit line
    fit_vals = reg_model.model(x_arr, y_arr, uncty_arr)
    plt.plot(x_arr, fit_vals, label=fit_type)

    # Display best-fit parameters, uncertainties, red_chisq
    params = ['equation', 'slope', 'intercept', 'red_chi_sq']
    pos_indices = param_pos_indices(params, reg_model)
    nonexistent_index = -1
    # Display Equation
    eq_index = pos_indices['equation']
    if eq_index != nonexistent_index:
        ref_eq_height = refactored_height(np.array([range_min, range_max]), eq_index)
        eq_description = f"Equation: {reg_model.equation}"
        print(eq_description)
        if show_ref:
            plt.text(min(x_arr), ref_eq_height, eq_description)
    # Display slope (m)
    m_index = pos_indices['slope']
    if m_index != nonexistent_index:
        ref_m_height = refactored_height(np.array([range_min, range_max]), m_index)
        m = reg_model.slope(x_arr, y_arr, uncty_arr)
        m_uncty = reg_model.slope_uncty(x_arr, y_arr, uncty_arr)
        if raw_data:
            m_equation = f"m = {m} \u00b1 {m_uncty}"
        else:
            m_equation = "m = %1.1f \u00b1 %1.1f" % (m, m_uncty)
        m_units = ""
        if y_units != "" and x_units != "":
            m_units = f"{y_units}/{x_units}"
        m_description = m_equation + " " + m_units
        print(m_description)
        if show_ref:
            plt.text(min(x_arr), ref_m_height, m_description)
    # Display intercept (c)
    c_index = pos_indices['intercept']
    if c_index != nonexistent_index:
        ref_c_height = refactored_height(np.array([range_min, range_max]), c_index)
        c = reg_model.intercept(x_arr, y_arr, uncty_arr)
        c_uncty = reg_model.intercept_uncty(x_arr, y_arr, uncty_arr)
        if raw_data:
            c_equation = f"c = {c} \u00b1 {c_uncty}"
        else:
            c_equation = "c = %1.1f \u00b1 %1.1f" % (c, c_uncty)
        c_units = ""
        if y_units != "" and x_units != "":
            c_units = f"{y_units}/{x_units}"
        c_description = c_equation + " " + c_units
        print(c_description)
        if show_ref:
            plt.text(min(x_arr), ref_c_height, c_description)
    # Display red_chisq
    red_chi_sq_index = pos_indices['red_chi_sq']
    if red_chi_sq_index != nonexistent_index:
        ref_red_chi_sq_height = refactored_height(np.array([range_min, range_max]), red_chi_sq_index)
        redchisq = red_chi_sq(x_arr, y_arr, uncty_arr, reg_model)
        if raw_data:
            print(f"Reduced Chi-Squared = {redchisq}")
        else:
            print("Reduced Chi-Squared = %1.1f" % redchisq)
        if show_ref:
            plt.text(min(x_arr), ref_red_chi_sq_height, "$\\tilde{\\chi}^2$ = %1.1f" % redchisq)
    plt.legend(frameon=True)
    plt.show()

def best_fit_plt_df(df: DataFrame, x_title: str, y_title: str, uncty_title: str, reg_model: type,
                    x_units="", y_units="", title="",
                    figsize=(6, 6), x_font_size=12, y_font_size=12,
                    title_font_size=14, x_tick_size=10, y_tick_size=10, grid=False, show_ref=False, raw_data=False):
    x_vals = df[x_title].values
    y_vals = df[y_title].values
    uncty_vals = df[uncty_title].values
    return best_fit_plt_arr(x_vals, y_vals, uncty_vals, reg_model,
                            x_title=x_title, y_title=y_title, x_units=x_units, y_units=y_units,
                            title=title, figsize=figsize,
                            x_font_size=x_font_size, y_font_size=y_font_size,
                            title_font_size=title_font_size, x_tick_size=x_tick_size,
                            y_tick_size=y_tick_size, grid=grid, show_ref=show_ref, raw_data=raw_data)

def refactored_height(arr_data, pos_index, start_height=390, sep_dist=30):
    assert 0 < start_height < 490, "0 < start_height < 490"
    axis_max = arr_data.max()
    axis_min = arr_data.min()
    total_height = axis_max-axis_min
    height_ratio = (start_height - sep_dist*pos_index)/490
    height_of_value = total_height * height_ratio
    dist_of_value_from_top = total_height - height_of_value
    refactored_height_of_val = axis_max - dist_of_value_from_top
    return refactored_height_of_val

def param_pos_indices(params: list, cls) -> dict:
    is_in_cls = [param in dir(cls) for param in params]
    index = 0
    lst = []
    for i in range(len(is_in_cls)):
        if (is_in_cls[i] == True) or (params[i] == "red_chi_sq"):
            lst += [index]
            index += 1
        else:
            lst += [-1]
    params_dict = {}
    for i in range(len(params)):
        params_dict[params[i]]=lst[i]
    return params_dict

def find_plot_range_extrema(arr, uncty_arr, min_or_max: str):
    assert min_or_max in ["min", "max"], "min_or_max must be 'min' or 'max'"
    testfpdre_df = pd.DataFrame({"vals":arr, "uncty":uncty_arr})
    vals_minus_uncty = "vals - uncty"
    vals_plus_uncty = "vals + uncty"
    testfpdre_df[vals_minus_uncty] = testfpdre_df["vals"]-testfpdre_df["uncty"]
    testfpdre_df[vals_plus_uncty] = testfpdre_df["vals"]+testfpdre_df["uncty"]
    if min_or_max == 'min':
        ascend = True
        col_filter = vals_minus_uncty
    else:
        ascend = False
        col_filter = vals_plus_uncty
    result = (testfpdre_df
    .sort_values(col_filter, ascending=ascend)
    .head(1)
    .reset_index()
    [col_filter]
    .values
    [0]
    )
    return result
#%%
