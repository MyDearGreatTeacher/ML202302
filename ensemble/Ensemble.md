# 集成式學習（Ensemble Learning）
- 非生成式演算法
  - 投票法（Voting）
  - 堆疊法（Stacking）
- 生成式演算法
  - 自助聚合法（Bootstrap Aggregation）
  - 提升法（Boosting）
    - 適應提升（Adaptive Boosting, AdaBoost）
    - 梯度提升（Gradient Boosting）
  - 隨機森林（Random Forest）

# [sklearn.ensemble 子模組](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble)

| Classifier | Regressor |
| ------ | -------|
| ensemble.VotingClassifier(estimators, `*[, ...]`) <br>Soft Voting/Majority Rule classifier for unfitted estimators.| ensemble.VotingRegressor(estimators, `*[, ...]`)<br>Prediction voting regressor for unfitted estimators.|
| ensemble.StackingClassifier(estimators[, ...])<br> Stack of estimators with a final classifier.| ensemble.StackingRegressor(estimators[, ...])<br> Stack of estimators with a final regressor.|
| ensemble.BaggingClassifier([estimator, ...])<br> A Bagging classifier.| ensemble.BaggingRegressor([estimator, ...])<br> A Bagging regressor.|
| ensemble.AdaBoostClassifier([estimator, ...]) <br> An AdaBoost classifier.| ensemble.AdaBoostRegressor([estimator, ...]) <br> An AdaBoost regressor.|
| ensemble.GradientBoostingClassifier(`*[, ...]`) <br>Gradient Boosting for classification.|ensemble.GradientBoostingRegressor(`*[, ...]`) <br>Gradient Boosting for regression.|
|ensemble.HistGradientBoostingClassifier([...])<br> Histogram-based Gradient Boosting Classification Tree.| ensemble.HistGradientBoostingRegressor([...]) <br>Histogram-based Gradient Boosting Regression Tree. |
| ensemble.RandomForestClassifier([...]) <br> A random forest classifier.| ensemble.RandomForestRegressor([...]) <br> A random forest regressor.| 
|  ensemble.ExtraTreesClassifier([...]) <br>An extra-trees classifier.| ensemble.ExtraTreesRegressor([n_estimators, ...])<br>  An extra-trees regressor.|
| ensemble.IsolationForest(`*[, n_estimators, ...]`) <br> Isolation Forest Algorithm.|||
|ensemble.RandomTreesEmbedding([...])<br> An ensemble of totally random trees.|||



# 參考書籍
- [Hands-On Ensemble Learning with Python](https://www.packtpub.com/product/hands-on-ensemble-learning-with-python/9781789612851) [GITHUB](https://github.com/PacktPublishing/Hands-On-Ensemble-Learning-with-Python)
  - 繁體中譯本[集成式學習：Python 實踐！整合全部技術，打造最強模型](https://www.tenlong.com.tw/products/9789863126942?list_name=srh)
- [Ensemble Machine Learning Cookbook(2019)](https://www.packtpub.com/product/ensemble-machine-learning-cookbook/9781789136609) [GITHUB](https://github.com/PacktPublishing/Ensemble-Machine-Learning-Cookbook)
