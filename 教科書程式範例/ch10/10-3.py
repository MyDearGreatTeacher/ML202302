df = pd.DataFrame(data = breast_cancer['data'], columns = breast_cancer['feature_names'])
df['target'] = breast_cancer['target']
df.head()