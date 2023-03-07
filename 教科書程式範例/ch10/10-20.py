from sklearn.tree import export_graphviz
import pydot 
from IPython.display import Image  

features = X.columns
class_names = ['惡性', '良性']
dot_data = export_graphviz(model_tree, out_file=None,
                           feature_names=features,
                           class_names = class_names,
                           proportion = False,
                           max_depth=3,
                           filled=True,
                           rounded=True)

graph = pydot.graph_from_dot_data(dot_data)  
graph[0].write_png('tumor.png')
Image(graph[0].create_png(), width=800)