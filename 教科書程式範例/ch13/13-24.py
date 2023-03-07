from sklearn.metrics import precision_recall_curve
fpr, tpr, thresholds = precision_recall_curve(y_test, 
                                              y_pred_proba)
plt.plot(fpr, tpr);