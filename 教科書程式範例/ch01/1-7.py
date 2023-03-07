list_k_cost = [30000, 500, 45000, 50000]
list_results = []
for k_cost in list_k_cost:
    nt_cost, nt_price = cal_price(k_cost)
    list_results.append([k_cost, nt_cost, nt_price])
list_results