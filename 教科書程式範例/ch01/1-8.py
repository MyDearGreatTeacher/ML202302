def cal_price_dict(k_cost):
    rate = 0.03
    nt_cost = k_cost * rate
    inc = 0.2
    nt_price = nt_cost * (1 + inc)
    data = {
        'k_cost': k_cost, 
        'nt_cost': nt_cost, 
        'nt_price': nt_price
    }
    return data