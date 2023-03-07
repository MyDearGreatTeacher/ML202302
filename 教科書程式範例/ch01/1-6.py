list_k_cost = [30000, 500, 45000, 50000]
for k_cost in list_k_cost:
    nt_cost, nt_price = cal_price(k_cost)
    print(f'韓國進貨成本是{k_cost}，台幣成本是{nt_cost}，台幣售價是{nt_price:.0f}元')