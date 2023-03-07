# 你如何知道管道器裡的轉換器名稱呢？make_pipeline會自動小寫轉換器的名稱當索引鍵。
# 如果還是不確定就用named_steps.keys()列出所有的索引鍵值
data_pl.named_transformers_['cat_pl'].named_steps.keys()