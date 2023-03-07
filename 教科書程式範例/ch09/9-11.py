from sklearn.tree import export_graphviz
import pydot 
from IPython.display import Image  
# features變數存放所有欄位名稱
features = cols
# class_names變數存放目標值表呈現的文字意義
class_names = ['死', '活']
# export_graphviz的第一個參數是決策樹模型的預測結果
# max_depth=3可設定決策樹呈現的深度，其餘參數讀者可自己測試
dot_data = export_graphviz(
    model_pl_tree.named_steps['decisiontreeclassifier'], 
    out_file=None,
    feature_names=features,
    class_names = class_names,
    proportion = False,
    max_depth=3,
    filled=True,
    rounded=True
)
graph = pydot.graph_from_dot_data(dot_data)  
# 也將結果存到tree.png檔案裡
graph[0].write_png('tree.png')
Image(graph[0].create_png())