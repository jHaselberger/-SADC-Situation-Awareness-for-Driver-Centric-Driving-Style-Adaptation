import numpy as np
from scipy.optimize import fmin


EPSILON = 1e-6


def pf_road(y, road_width, A_road=1.0):
    u_right = A_road / (0.5 * road_width - y + EPSILON)
    u_left = A_road / (0.5 * road_width + y + EPSILON)

    u_road = u_right + u_left
    return u_road


def pf_lane(y, y_0=0.0, A_lane=1.0):
    u_lane = A_lane * np.power(y - y_0, 2)
    return u_lane


def pf_ccg(y, a_y, ccg=0.1, A_ccg=1.0, a_y_max=5.0):
    d_ccg = a_y * ccg
    u_ccg = A_ccg * np.power(y + d_ccg, 2)

    if type(u_ccg) == np.float64:
        if d_ccg >= 0.0:
            if y + d_ccg <= 0.0:
                u_ccg = 0.0
        else:
            if y + d_ccg > 0.0:
                u_ccg = 0.0
    else:
        if d_ccg >= 0.0:
            u_ccg[y + d_ccg < 0.0] = 0.0
        else:
            u_ccg[y + d_ccg > 0.0] = 0.0

    u_ccg = abs(a_y) / (a_y_max + EPSILON) * u_ccg 

    return u_ccg


def pf_oncoming_car(y, ttc, d_y_traffic=0.0, A_ttc=1.0, a_ttc_max=20, ttc_max=8):
    _d_y_traffic = (-d_y_traffic / ttc_max) * ttc + d_y_traffic
    u_ttc_y = A_ttc * np.power(y - _d_y_traffic, 2)

    if y >= _d_y_traffic:
        u_ttc_y = 0.0

    a_ttc = (-a_ttc_max / ttc_max) * ttc + a_ttc_max

    if ttc <= 0.0:
        a_ttc = 0.0
    else:
        a_ttc = np.maximum(a_ttc, 0.0)

    return u_ttc_y * a_ttc


def get_u(
    y,
    ttc,
    a_y,
    A_ttc=1.0,
    A_road=1.0,
    A_lane=1.0,
    A_ccg=1.0,
    d_y_traffic=0.0,
    a_ttc_max=20,
    ttc_max=8,
    y_0=0.0,
    w=3.0,
    ccg=0.1,
    a_y_max=5.0,
):
    u_ttc = pf_oncoming_car(
        y=y,
        ttc=ttc,
        a_ttc_max=a_ttc_max,
        d_y_traffic=d_y_traffic,
        A_ttc=A_ttc,
        ttc_max=ttc_max,
    )
    u_road = pf_road(y, road_width=w, A_road=A_road)
    u_lane = pf_lane(y, y_0=y_0, A_lane=A_lane)
    u_ccg = pf_ccg(y, a_y, ccg=ccg, A_ccg=A_ccg, a_y_max=a_y_max)

    u = u_ttc + u_road + u_lane + u_ccg
    return u


def get_best_y(
    ttc,
    a_y,
    A_ttc,
    A_road,
    A_lane,
    A_ccg,
    d_y_traffic,
    a_ttc_max,
    ttc_max,
    y_0,
    w,
    ccg,
    a_y_max,
):
    r = fmin(
        get_u,
        0.0,
        disp=False,
        args=(
            ttc,
            a_y,
            A_ttc,
            A_road,
            A_lane,
            A_ccg,
            d_y_traffic,
            a_ttc_max,
            ttc_max,
            y_0,
            w,
            ccg,
            a_y_max,
        ),
    )
    return r[0]


