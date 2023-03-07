from sklearn.metrics import roc_curve, roc_auc_score
y_pred_proba = model_pl_rf.predict_proba(X_test)[:,1] 
fpr, tpr, thresholds = roc_curve(y_test, 
                                 y_pred_proba)
plt.plot(fpr, tpr)
plt.xlim(-0.01,1)
plt.ylim(0,1.01)
plt.plot([0,1],[0,1], ls='--')
roc_auc_score(y_test, model_pl_rf.predict_proba(X_test)[:,1])