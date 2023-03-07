pd.concat([df['Survived'].value_counts(),
          df['Survived'].value_counts(normalize=True)], 
          axis=1, keys=['個數','百分比'])