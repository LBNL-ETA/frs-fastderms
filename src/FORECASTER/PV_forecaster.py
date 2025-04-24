"""
Part of the FAST-DERMS Flexible Resource Scheduler Demonstration
"""

import csv
import pickle
import time
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, Exponentiation
from scipy.stats import norm


def train_PV_forecaster(num_PV_train, path_to_PV_data, path_to_cal_temp_data):
    # ---- read PV generation data for training and convert to a 2D matrix ------
    with open(path_to_PV_data, "r") as f:
        acco_G_in_row_train = list(csv.reader(f, delimiter=","))

    acco_G_in_row_train = np.array(acco_G_in_row_train).astype("float")

    # ---- normalize the generations ------
    temp = acco_G_in_row_train[0:num_PV_train, 1:]
    aggr_G_train = temp.sum(axis=0)
    hr_num_train = len(aggr_G_train)
    G_peak_train = np.amax(aggr_G_train)
    aggr_G_train_nrm = np.divide(aggr_G_train, G_peak_train)
    aggr_G_train_nrm = np.transpose(aggr_G_train_nrm)
    aggr_G_train_nrm = np.reshape(aggr_G_train_nrm, (hr_num_train, -1))

    # ---- read calendar and temperature variables, convert to a 2D matrix ------
    with open(path_to_cal_temp_data, "r") as f:
        date_Hd_Dy_GHI_train = list(csv.reader(f, delimiter=","))

    date_Hd_Dy_GHI_train = np.array(date_Hd_Dy_GHI_train).astype("float")

    hr_in_dy = date_Hd_Dy_GHI_train[0:, 5]
    dy_in_yr = date_Hd_Dy_GHI_train[0:, 6]
    GHI_train = date_Hd_Dy_GHI_train[0:, 7]
    hr_in_dy = np.reshape(hr_in_dy, (hr_num_train, -1))
    dy_in_yr = np.reshape(dy_in_yr, (hr_num_train, -1))
    GHI_train = np.reshape(GHI_train, (hr_num_train, -1))
    hr_in_dy_nrm = np.divide(hr_in_dy, np.amax(hr_in_dy))
    dy_in_yr_nrm = np.divide(dy_in_yr, np.amax(dy_in_yr))
    GHI_train_nrm = np.divide(GHI_train, np.amax(GHI_train))

    # --------- construct the training dataset: X - input , Y - output (G) -------
    X_train_nrm = np.concatenate((hr_in_dy_nrm, dy_in_yr_nrm, GHI_train_nrm), axis=1)
    Y_train_nrm = aggr_G_train_nrm

    # training
    kernel = Exponentiation(RationalQuadratic(), exponent=2)
    gprMdl_nrm_trained = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(
        X_train_nrm, Y_train_nrm
    )

    path_to_data = Path(path_to_PV_data).parent
    filename = path_to_data / "gprMdl_nrm_trained_G.sav"

    with open(filename.absolute(), "wb") as f:
        pickle.dump(gprMdl_nrm_trained, f)

    return X_train_nrm, Y_train_nrm


def PV_prediction(
    idx_PV_predict,
    day,
    path_to_training_data,
    path_to_predict_data,
    path_to_cal_temp_data,
):
    print("Predicting the PV generation on the " + str(day) + "th day...")
    start = time.time()

    with open(path_to_training_data, "r") as f:
        acco_G_in_row_train = list(csv.reader(f, delimiter=","))
    acco_G_in_row_train = np.array(acco_G_in_row_train).astype("float")

    with open(path_to_predict_data, "r") as f:
        acco_G_in_row_predict = list(csv.reader(f, delimiter=","))
    acco_G_in_row_predict = np.array(acco_G_in_row_predict).astype("float")

    hr_num_predict = len(acco_G_in_row_predict[0, 1:])

    # time and temperature variables
    with open(path_to_cal_temp_data, "r") as f:
        date_Hd_Dy_GHI_predict = list(csv.reader(f, delimiter=","))
    date_Hd_Dy_GHI_predict = np.array(date_Hd_Dy_GHI_predict).astype("float")

    hr_in_dy = date_Hd_Dy_GHI_predict[0:, 5]
    dy_in_yr = date_Hd_Dy_GHI_predict[0:, 6]
    GHI_predict = date_Hd_Dy_GHI_predict[0:, 7]
    hr_in_dy = np.reshape(hr_in_dy, (hr_num_predict, -1))
    dy_in_yr = np.reshape(dy_in_yr, (hr_num_predict, -1))
    GHI_predict = np.reshape(GHI_predict, (hr_num_predict, -1))
    hr_in_dy_nrm = np.divide(hr_in_dy, np.amax(hr_in_dy))
    dy_in_yr_nrm = np.divide(dy_in_yr, np.amax(dy_in_yr))
    GHI_predict_nrm = np.divide(GHI_predict, np.amax(GHI_predict))

    num_PV_predict = len(idx_PV_predict)  #
    # normalize the loads in prediction dataset using the peak load in training dataset
    temp_train = acco_G_in_row_train[
        idx_PV_predict, 1:
    ]  # for estimating the future peak in prediction dataset
    aggr_G_train = temp_train.sum(axis=0)
    G_peak_predict_estimated = np.amax(aggr_G_train)
    temp = acco_G_in_row_predict[idx_PV_predict, 1:]
    aggr_G_predict = temp.sum(axis=0)
    aggr_G_predict_nrm = np.divide(aggr_G_predict, G_peak_predict_estimated)
    aggr_G_predict_nrm = np.transpose(aggr_G_predict_nrm.reshape(1, hr_num_predict))

    # build an array of last genertions for predicting
    temp = aggr_G_predict_nrm[0, 0]
    temp_1 = aggr_G_predict_nrm[0:-1, 0]
    last_G_predict_nrm = np.concatenate(
        (temp.reshape(1, 1), temp_1.reshape(hr_num_predict - 1, 1)), axis=0
    )

    PV_num_temp = 30
    # print(PV_num_temp)

    path_to_data = Path(path_to_training_data).parent
    filename = path_to_data / "gprMdl_nrm_trained_G.sav"

    with open(filename.absolute(), "rb") as f:
        trained_GPR_model = pickle.load(f)

    # specify the number of Monte Carlo samples in each iteration
    mon_carl_num = 15
    mu_sd_collect = np.array([], dtype=np.float32).reshape(0, 2)
    real_G_collect = np.array([], dtype=np.float32).reshape(0, 1)
    horizon = 24
    # print("start PV generation forecasting")
    for day_str in range(day, day + 1):
        mu_sd_collect_temp = np.array([], dtype=np.float32).reshape(0, 2)
        real_G_collect_temp = np.array([], dtype=np.float32).reshape(0, 1)
        indx = (day_str - 1) * 24 + 1
        np.random.seed(1)
        for j in range(horizon):
            # print('The ' + str(j + 1) + 'th hour')
            # start = time.time()
            if j == 0:
                last_G = last_G_predict_nrm[indx - 1]
            time_vari = [hr_in_dy_nrm[indx - 1], dy_in_yr_nrm[indx - 1]]
            time_vari = np.asarray(time_vari, dtype=np.float32)
            time_vari = np.transpose(time_vari)
            # print(time_vari)

            if np.remainder(j, 24) <= 6 or np.remainder(j, 24) >= 20:
                GHI_temp = np.zeros((len(last_G), 1))
            else:
                GHI_temp = GHI_predict_nrm[indx - 1, :] + np.random.normal(
                    0, 0.01, [len(last_G), 1]
                )
                GHI_temp = np.absolute(GHI_temp)

            last_G = np.array(last_G, ndmin=2)
            X_test_nrm = np.concatenate(
                (np.matlib.repmat(time_vari, len(last_G), 1), GHI_temp), axis=1
            )
            # print(X_test_nrm)
            #  print(trained_GPR_model.predict(X_test_nrm, return_std=True))

            predicted_mu_sd_temp = np.empty(shape=(len(last_G), 2))
            for k in range(len(last_G)):
                # print(X_test_nrm[k, 0:])
                predicted_mu_sd_temp[k, 0], predicted_mu_sd_temp[k, 1] = (
                    trained_GPR_model.predict(
                        np.array(X_test_nrm[k, 0:], ndmin=2), return_std=True
                    )
                )

            # update PV generation at the last time point
            if j == 0:
                mu_sd_collect_temp = np.vstack(
                    [mu_sd_collect_temp, predicted_mu_sd_temp]
                )
                last_G = np.random.normal(
                    predicted_mu_sd_temp[0, 0],
                    predicted_mu_sd_temp[0, 1],
                    [mon_carl_num, 1],
                )
            else:
                last_G = np.empty(shape=(mon_carl_num, 1))
                for kk in range(mon_carl_num):
                    last_G[kk, 0] = np.random.normal(
                        predicted_mu_sd_temp[kk, 0], predicted_mu_sd_temp[kk, 1], [1, 1]
                    )

                phat = norm.fit(last_G)
                phat = np.array(phat).reshape(1, 2)
                mu_sd_collect_temp = np.vstack([mu_sd_collect_temp, phat])

            if np.remainder(j, 24) <= 6 or np.remainder(j, 24) >= 20:
                mu_sd_collect_temp[j, :] = 0

            # end = time.time()
            # print(f"Runtime for this hour is {end - start}" + " seconds.")

            real_G_collect_temp = np.concatenate(
                (
                    real_G_collect_temp,
                    np.array(aggr_G_predict_nrm[indx - 1, 0], ndmin=2),
                ),
                axis=0,
            )
            indx = indx + 1

        mu_sd_collect = np.concatenate((mu_sd_collect, mu_sd_collect_temp), axis=0)
        real_G_collect = np.concatenate((real_G_collect, real_G_collect_temp), axis=0)

    mu_sd_collect = np.multiply(mu_sd_collect, G_peak_predict_estimated)
    real_G_collect = np.multiply(real_G_collect, G_peak_predict_estimated)
    mu_std_realG = np.concatenate((mu_sd_collect, real_G_collect), axis=1)
    end = time.time()
    print(f"Runtime for this day is {end - start}" + " seconds.")
    return mu_std_realG
