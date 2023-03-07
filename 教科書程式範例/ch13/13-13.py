df_left_time = df.groupby(['left','time_spend_company']).size().unstack(0)
df_left_time.plot(kind='bar');