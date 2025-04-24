"""
Part of the FAST-DERMS Flexible Resource Scheduler Demonstration
"""

import csv
import pickle
import time

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, Exponentiation
from scipy.stats import norm
from pathlib import Path


def train_load_forecaster(num_cust_train, path_to_conso_data, path_to_cal_temp_data):
    # ---- read residential customers' load data and convert to a 2D matrix ------
    with open(path_to_conso_data, "r") as f:
        acco_P_in_row_resi_train = list(csv.reader(f, delimiter=","))

    acco_P_in_row_train = np.array(acco_P_in_row_resi_train).astype("float")

    # ---- normalize the loads ------
    temp = acco_P_in_row_train[0:num_cust_train, 1:]
    aggr_P_train = temp.sum(axis=0)
    hr_num_train = len(aggr_P_train)
    P_peak_train = np.amax(aggr_P_train)
    aggr_P_train_nrm = np.divide(aggr_P_train, P_peak_train)
    aggr_P_train_nrm = np.transpose(aggr_P_train_nrm)
    aggr_P_train_nrm = np.reshape(aggr_P_train_nrm, (hr_num_train, -1))

    # ---  create the input variable of the load at (t-1)  ---
    temp = aggr_P_train_nrm[0, 0]
    temp_1 = aggr_P_train_nrm[0:-1, 0]
    last_aggr_P_train_nrm = np.concatenate(
        (temp.reshape(1, 1), temp_1.reshape(hr_num_train - 1, 1)), axis=0
    )

    # ---- read calendar and temperature variables, convert to a 2D matrix ------
    with open(path_to_cal_temp_data, "r") as f:
        date_Hd_Dw_Dy_My_T_train = list(csv.reader(f, delimiter=","))

    date_Hd_Dw_Dy_My_T_train = np.array(date_Hd_Dw_Dy_My_T_train).astype("float")
    hr_in_dy = date_Hd_Dw_Dy_My_T_train[0:, 5]
    T_train = date_Hd_Dw_Dy_My_T_train[0:, 9]
    hr_in_dy = np.reshape(hr_in_dy, (hr_num_train, -1))
    T_train = np.reshape(T_train, (hr_num_train, -1))

    # --------- construct the training dataset: X - input , Y - output (P) -------
    X_train_nrm = np.concatenate((hr_in_dy, T_train, last_aggr_P_train_nrm), axis=1)
    print(np.shape(X_train_nrm))

    Y_train_nrm = aggr_P_train_nrm

    # training
    kernel = Exponentiation(RationalQuadratic(), exponent=2)
    gprMdl_nrm_trained = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(
        X_train_nrm, Y_train_nrm
    )
    path_to_data = Path(path_to_conso_data).parent
    filename = path_to_data / "gprMdl_nrm_trained_P.sav"

    with open(filename.absolute(), "wb") as f:
        pickle.dump(gprMdl_nrm_trained, f)


def load_prediction(
    idx_cust_predict,
    day,
    path_to_training_data,
    path_to_predict_data,
    path_to_cal_temp_data,
):
    print("Predicting the load on the " + str(day) + "th day...")
    start = time.time()

    # ----- load csv files  --------
    # start = time.time()
    with open(path_to_training_data, "r") as f:
        acco_P_in_row_resi_train = list(csv.reader(f, delimiter=","))
    acco_P_in_row_train = np.array(acco_P_in_row_resi_train).astype("float")

    with open(path_to_predict_data, "r") as f:
        acco_P_in_row_resi_predcit = list(csv.reader(f, delimiter=","))
    acco_P_in_row_predict = np.array(acco_P_in_row_resi_predcit).astype("float")

    hr_num_predict = len(acco_P_in_row_predict[0, 1:])
    # print(hr_num_predict)

    # time and temperature variables
    with open(path_to_cal_temp_data, "r") as f:
        date_Hd_Dw_Dy_My_T_predict = list(csv.reader(f, delimiter=","))
    date_Hd_Dw_Dy_My_T_predict = np.array(date_Hd_Dw_Dy_My_T_predict).astype("float")
    hr_in_dy = date_Hd_Dw_Dy_My_T_predict[0:, 5]
    T_predict = date_Hd_Dw_Dy_My_T_predict[0:, 9]
    hr_in_dy = np.reshape(hr_in_dy, (hr_num_predict, -1))
    T_predict = np.reshape(T_predict, (hr_num_predict, -1))

    num_cust_predict = len(idx_cust_predict)  #
    # normalize the loads in prediction dataset using the peak load in training dataset
    temp_train = acco_P_in_row_train[
        idx_cust_predict, 1:
    ]  # for estimating the future peak in prediction dataset
    aggr_P_train = temp_train.sum(axis=0)
    P_peak_predict_estimated = np.amax(aggr_P_train)
    # print(P_peak_predict_estimated)
    temp = acco_P_in_row_predict[idx_cust_predict, 1:]
    aggr_P_predict = temp.sum(axis=0)
    # print(np.max(aggr_P_predict))
    aggr_P_predict_nrm = np.divide(aggr_P_predict, P_peak_predict_estimated)
    aggr_P_predict_nrm = np.transpose(aggr_P_predict_nrm.reshape(1, hr_num_predict))

    # build an array of last loads for predicting
    temp = aggr_P_predict_nrm[0, 0]
    temp_1 = aggr_P_predict_nrm[0:-1, 0]
    last_P_predict_nrm = np.concatenate(
        (temp.reshape(1, 1), temp_1.reshape(hr_num_predict - 1, 1)), axis=0
    )

    path_to_data = Path(path_to_training_data).parent
    filename = path_to_data / "gprMdl_nrm_trained_P.sav"

    with open(filename.absolute(), "rb") as f:
        trained_GPR_model = pickle.load(f)

    # specify the number of Monte Carlo samples in each iteration
    mon_carl_num = 15
    mu_sd_collect = np.array([], dtype=np.float32).reshape(0, 2)
    real_P_collect = np.array([], dtype=np.float32).reshape(0, 1)
    horizon = 24
    # print("start load forecasting")
    for day_str in range(day, day + 1):
        mu_sd_collect_temp = np.array([], dtype=np.float32).reshape(0, 2)
        real_P_collect_temp = np.array([], dtype=np.float32).reshape(0, 1)
        indx = (day_str - 1) * 24 + 1
        np.random.seed(1)
        for j in range(horizon):
            # print('The ' + str(j + 1) + 'th hour')
            # start = time.time()
            if j == 0:
                last_P = last_P_predict_nrm[indx - 1]

            time_vari = hr_in_dy[indx - 1]
            time_vari = np.asarray(time_vari, dtype=np.float32)
            # time_vari = np.transpose(time_vari)
            T_temp = T_predict[indx - 1, :] + np.random.normal(0, 1, [len(last_P), 1])
            last_P = np.array(last_P, ndmin=2)
            X_test_nrm = np.concatenate(
                (np.broadcast_to(time_vari, (len(last_P), 1)), T_temp, last_P), axis=1
            )
            # print(X_test_nrm)
            predicted_mu_sd_temp = np.empty(shape=(len(last_P), 2))

            for k in range(len(last_P)):
                predicted_mu_sd_temp[k, 0], predicted_mu_sd_temp[k, 1] = (
                    trained_GPR_model.predict(
                        np.array(X_test_nrm[k, 0:], ndmin=2), return_std=True
                    )
                )

            # update load at the last time point
            if j == 0:
                mu_sd_collect_temp = np.vstack(
                    [mu_sd_collect_temp, predicted_mu_sd_temp]
                )
                last_P = np.random.normal(
                    predicted_mu_sd_temp[0, 0],
                    predicted_mu_sd_temp[0, 1],
                    [mon_carl_num, 1],
                )
            else:
                last_P = np.empty(shape=(mon_carl_num, 1))
                for kk in range(mon_carl_num):
                    last_P[kk, 0] = np.random.normal(
                        predicted_mu_sd_temp[kk, 0], predicted_mu_sd_temp[kk, 1], [1, 1]
                    )

                phat = norm.fit(last_P)
                phat = np.array(phat).reshape(1, 2)
                mu_sd_collect_temp = np.vstack([mu_sd_collect_temp, phat])
            # end = time.time()
            # print(f"Runtime for this hour is {end - start}" + " seconds.")

            real_P_collect_temp = np.concatenate(
                (
                    real_P_collect_temp,
                    np.array(aggr_P_predict_nrm[indx - 1, 0], ndmin=2),
                ),
                axis=0,
            )
            indx = indx + 1

        mu_sd_collect = np.concatenate((mu_sd_collect, mu_sd_collect_temp), axis=0)
        real_P_collect = np.concatenate((real_P_collect, real_P_collect_temp), axis=0)

    mu_sd_collect = np.multiply(mu_sd_collect, P_peak_predict_estimated)
    real_P_collect = np.multiply(real_P_collect, P_peak_predict_estimated)
    mu_std_realP = np.concatenate((mu_sd_collect, real_P_collect), axis=1)
    end = time.time()
    print(f"Runtime for this day is {end - start}" + " seconds.")
    return mu_std_realP


if __name__ == "__main__":
    # main function
    num_cust_for_trianing = 100  # specify the number of customers whose loads are used for training a general forecaster
    train_forecaster(num_cust_for_trianing)  # train a general forecaster
