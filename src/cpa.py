import numpy as np


def linear_fit(x, y):
    N_ = len(x)
    y_ = np.array(y)
    X_ = np.vstack([x, np.ones(N_)]).T

    fit_results = np.linalg.lstsq(X_, y_)

    a_, b_ = fit_results[0]
    r2error_ = fit_results[1][0]

    return a_, b_, r2error_


def cumsum_diff(residual):
    cumsum = np.cumsum(residual)

    return np.max(cumsum) - np.min(cumsum)


def permutation_test(residual, n=1000):

    permuted_cumdiff_list = []
    for i in range(0, n):
        permuted_residual = np.random.permutation(residual)
        permuted_cumdiff = cumsum_diff(permuted_residual)

        permuted_cumdiff_list.append(permuted_cumdiff)

    return permuted_cumdiff_list, n


def piece_wise_linear_fit(x, y, division_point):
    left_slope, _, left_r2error = linear_fit(x[0:division_point + 1], y[0:division_point + 1])
    right_slope, _, right_r2error = linear_fit(x[division_point:], y[division_point:])

    return left_slope, left_r2error, right_slope, right_r2error


def check_cp_likelihood(residual):

    original_cumdiff = cumsum_diff(residual)

    permuted_cumdiff_list, list_len = permutation_test(residual)

    cp_likelihood = np.sum(permuted_cumdiff_list < original_cumdiff) / float(list_len)

    return cp_likelihood


def detect_a_cp(x, y):
    data_len = len(x)

    results_dict = {}
    error_sum_list = []
    for i in range(2, data_len-2):
        left_slope, left_r2error, right_slope, right_r2error = piece_wise_linear_fit(x, y, i)

        results_dict[i] = ((left_slope, right_slope), (left_r2error, right_r2error))
        error_sum_list.append(left_r2error + right_r2error)

    optimal_index = np.argmin(error_sum_list)+2
    fit_results = results_dict[optimal_index]

    return optimal_index, fit_results


def detect_all_cp(x, y, detection_threshold=0.05):
    cp_list = []

    a, b, r2error = linear_fit(x, y)
    residual = y - (a * x + b)

    cp_likelihood = check_cp_likelihood(residual)

    if cp_likelihood > 1.0 - detection_threshold:
        cp_index, _ = detect_a_cp(x, y)

        cp_list.append(x[cp_index])

        if cp_index > 5:
            x_left, y_left = x[0:cp_index + 1], y[0:cp_index + 1]
            cp_list += detect_all_cp(x_left, y_left)

        if cp_index < len(x)-5:
            x_right, y_right = x[cp_index:], y[cp_index:]
            cp_list += detect_all_cp(x_right, y_right)

    return cp_list


class ChangePointAnalysis():

    def __init__(self, detection_threshold=0.05):
        self.detection_threshold = detection_threshold
        self.cp_list = np.array([])

    def run(self, x, y):
        cp_list = detect_all_cp(x, y, detection_threshold=self.detection_threshold)

        self.cp_list = np.sort(cp_list)

    def get_result(self):
        return self.cp_list


class PieceWiseLinearFit():

    def __init__(self, detection_threshold=0.05):
        self.detection_threshold = detection_threshold

        self.change_point = None
        self.cp_likelihood = None
        self.slopes = None
        self.r2errors = None

    def fit(self, x, y):
        a, b, r2error = linear_fit(x, y)
        residual = y - (a * x + b)

        cp_index, slopes_and_errors = detect_a_cp(x, y)

        self.change_point = x[cp_index]
        self.cp_likelihood = check_cp_likelihood(residual)
        self.slopes = slopes_and_errors[0]
        self.r2errors = slopes_and_errors[1]

# class MultiPieceWiseLinearFit():
#
#     def __init__(self, division_num, detection_threshold=0.05):
#         self.detection_threshold = detection_threshold
#         self.div_num = division_num
#
#     def fit(self, x, y):
#         fit_results_dict = {}
#
#         for i in range(0, self.div_num):
#             a, b, r2error = linear_fit(x, y)
#             residual = y - (a * x + b)
#
#             cp_likelihood = check_cp_likelihood(residual)
#
#             cp_index, fit_results = detect_a_cp(x, y)
#             [x[cp_index], cp_likelihood, fit_results]