k_cost = 50000
rate = 0.03
nt_cost = k_cost * rate
inc = 0.2
nt_price = nt_cost * (1 + inc)
s = f'韓國進貨成本是{k_cost}，台幣成本是{nt_cost}，台幣售價是{nt_price:.0f}元'
print(s)