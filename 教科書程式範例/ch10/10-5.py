print(f'標籤0為{breast_cancer["target_names"][0]}，是惡性腫瘤的意思')
print(df['target'].value_counts(normalize=True))