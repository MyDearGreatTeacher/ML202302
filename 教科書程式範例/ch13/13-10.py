df_left_salary = df.groupby(['left','salary']).size().unstack(1)
df_left_salary = df_left_salary[['low', 'medium', 'high']]
df_left_salary