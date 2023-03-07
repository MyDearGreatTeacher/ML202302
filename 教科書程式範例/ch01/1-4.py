def cal_price(k_cost):
    rate = 0.03
    nt_cost = k_cost * rate
    inc = 0.2
    nt_price = nt_cost * (1 + inc)
    return nt_cost, nt_price