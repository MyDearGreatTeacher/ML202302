df.groupby(['left','sales']).size().unstack(0).plot(kind='bar', stacked=True);