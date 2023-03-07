list_k_cost = [30000, 500, 45000, 50000]
list_results = []
for k_cost in list_k_cost:
    if k_cost > 10000:
        data = cal_price_dict(k_cost)
        list_results.append(data)
list_results