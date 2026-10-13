# 🎓 VIVA QUESTIONS & ANSWERS
## Dual Disease Prediction System — ML CA2
### Adil Khan (12510526) | Bhuvan Pratap Singh (12515658)

---

## SECTION 1: PROJECT BASICS

**Q1. What is your project about?**
> Our project is a Dual Disease Prediction System that uses Machine Learning to predict two diseases — Diabetes and Breast Cancer — in a single unified platform. Users enter medical values and the system predicts whether the patient is at risk, using the best automatically-selected ML model.

**Q2. Why did you choose this topic?**
> These are two of the most prevalent diseases globally. India has over 77 million diabetic patients and breast cancer is the most common cancer in women. Early detection significantly improves treatment outcomes. ML can assist doctors in early-stage screening, making our system practically impactful.

**Q3. What makes your project unique/novel?**
> Most student projects predict only one disease. Our system:
> - Handles two diseases in one platform
> - Compares 6 ML algorithms side by side
> - Auto-selects the best model per disease
> - Provides a real-time web app via Streamlit
> - Includes PCA, ROC curves, and full EDA visualizations

---

## SECTION 2: DATASETS

**Q4. Which datasets did you use?**
> 1. **Pima Indians Diabetes Dataset** — NIDDK/UCI, 768 samples, 8 features
> 2. **Wisconsin Breast Cancer Dataset** — UCI ML Repository, 569 samples, 30 features

**Q5. What preprocessing did you do for diabetes data?**
> We replaced zero values in biologically impossible columns (Glucose, BloodPressure, SkinThickness, Insulin, BMI) with column medians, since a human cannot have 0 glucose or 0 blood pressure. Then we applied StandardScaler normalization.

**Q6. What is the class distribution in both datasets?**
> - Diabetes: 65.1% Not Diabetic, 34.9% Diabetic (imbalanced)
> - Cancer: 62.7% Benign, 37.3% Malignant (slight imbalance)

**Q7. What is the train-test split you used?**
> 80-20 split with `random_state=42` and `stratify=y` to maintain class proportion in both sets.

---

## SECTION 3: DATA PREPROCESSING

**Q8. What is StandardScaler? Why did you use it?**
> StandardScaler transforms features to have mean=0 and standard deviation=1. We used it because algorithms like KNN, SVM, and Logistic Regression are sensitive to feature scale. Without scaling, a feature like Insulin (0–850) would dominate over Pregnancies (0–17).

**Q9. What is the formula for StandardScaler?**
> z = (x − μ) / σ  where μ is the mean and σ is the standard deviation of the feature.

**Q10. Why did you replace zeros with median and not mean?**
> Median is more robust to outliers than mean. In medical data with extreme outliers (e.g., Insulin has values up to 846), using median prevents skewing the replacement value.

---

## SECTION 4: MACHINE LEARNING MODELS

**Q11. Explain Logistic Regression.**
> Logistic Regression is a linear classifier that uses the sigmoid function to output probabilities between 0 and 1. It models the log-odds of the outcome as a linear combination of input features. Best for linearly separable data and very interpretable.

**Q12. What is the sigmoid function?**
> σ(z) = 1 / (1 + e^(-z))  where z = wᵀx + b. It squishes any real number into the range (0, 1), giving us a probability.

**Q13. Explain Random Forest.**
> Random Forest is an ensemble of multiple Decision Trees trained on different bootstrap samples (random data subsets) using a random subset of features at each split. Final prediction is majority vote. It reduces overfitting and is one of the most accurate general-purpose algorithms.

**Q14. What is bagging vs boosting?**
> - **Bagging** (Bootstrap Aggregating): Trees trained independently, results averaged. Random Forest uses bagging.
> - **Boosting**: Trees trained sequentially, each correcting the previous one's errors. Gradient Boosting uses boosting.

**Q15. What is KNN? How does it work?**
> K-Nearest Neighbors classifies a new point by looking at its K closest training points (using Euclidean distance) and taking the majority class. We used K=5. It's lazy learning — no training phase, just memorizes data.

**Q16. Explain Naive Bayes.**
> Naive Bayes applies Bayes' Theorem with the "naive" assumption that all features are conditionally independent. P(class | features) ∝ P(features | class) × P(class). It's very fast and works well even on small datasets.

**Q17. What is SVM? What kernel did you use?**
> SVM (Support Vector Machine) finds the optimal hyperplane that maximizes the margin between classes. We used the RBF (Radial Basis Function) kernel which handles non-linear boundaries by mapping data to higher dimensions.

---

## SECTION 5: MODEL EVALUATION

**Q18. What evaluation metrics did you use?**
> - **Accuracy**: (TP + TN) / Total
> - **Precision**: TP / (TP + FP) — minimizes false alarms
> - **Recall**: TP / (TP + FN) — minimizes missed cases (critical in medicine)
> - **F1-Score**: Harmonic mean of Precision and Recall
> - **ROC-AUC**: Area under the ROC curve (best for imbalanced datasets)
> - **Cross-Validation**: 5-fold CV to test generalization

**Q19. What is a Confusion Matrix?**
> A 2×2 table showing:
> - TP (True Positive): Correctly predicted positive
> - TN (True Negative): Correctly predicted negative
> - FP (False Positive): Predicted positive but actually negative (Type I error)
> - FN (False Negative): Predicted negative but actually positive (Type II error — most dangerous in medicine)

**Q20. Why is Recall more important than Precision in medical prediction?**
> A False Negative (predicting "no disease" when disease is present) is far more dangerous than a False Positive. Missing a diabetic or cancer patient can be life-threatening. So high Recall is prioritized in medical applications.

**Q21. What is ROC curve and AUC?**
> ROC (Receiver Operating Characteristic) curve plots True Positive Rate vs False Positive Rate at various thresholds. AUC (Area Under Curve) summarizes this into one number. AUC = 1.0 is perfect; AUC = 0.5 is random guessing. It's especially useful for imbalanced datasets.

**Q22. What is cross-validation? Why use 5-fold CV?**
> Cross-validation splits training data into K folds. The model trains on K-1 folds and validates on the remaining fold, repeated K times. We use 5-fold to get a reliable estimate of how the model generalizes to unseen data, reducing variance in accuracy estimates.

---

## SECTION 6: PCA

**Q23. What is PCA?**
> Principal Component Analysis is an unsupervised dimensionality reduction technique. It transforms features into a new set of uncorrelated components (Principal Components) ordered by the amount of variance they explain.

**Q24. Why did you use PCA?**
> The Breast Cancer dataset has 30 features, which is hard to visualize. PCA helped us reduce it to 2D for visualization, while retaining the most important variance structure. It confirmed that the two classes are separable, justifying our ML approach.

**Q25. What is explained variance?**
> It's the proportion of total dataset variance captured by each principal component. If PC1 explains 60% and PC2 explains 20%, together they capture 80% of the information in a 2D plot.

---

## SECTION 7: WEB APP

**Q26. What technology did you use for the web app?**
> Streamlit — a Python library that turns Python scripts into interactive web applications without HTML/CSS/JavaScript coding. It's perfect for ML demos.

**Q27. How do you run the web app?**
> ```bash
> pip install streamlit
> streamlit run app/streamlit_app.py
> ```
> It opens automatically at http://localhost:8501

**Q28. How are models saved and loaded?**
> Using `joblib.dump()` to save and `joblib.load()` to load. Trained models are saved as `.pkl` files in the `models/` folder. This allows the web app to load pre-trained models instantly without retraining.

---

## SECTION 8: RESULTS

**Q29. Which model performed best for Diabetes and why?**
> Random Forest achieved the highest accuracy (~78%) for diabetes because it reduces variance through ensemble averaging, handles the moderate imbalance well, and captures non-linear feature interactions that logistic regression misses.

**Q30. Which model performed best for Breast Cancer and why?**
> SVM and Random Forest both performed well (~97% accuracy). SVM excels here because the 30-dimensional feature space is ideal for kernel-based separation, and the RBF kernel handles the non-linear boundaries effectively.

**Q31. What is the most important feature for diabetes prediction?**
> Glucose level is the most important feature (importance ~24%). Clinically, this makes sense — elevated blood glucose is the primary diagnostic marker for diabetes. BMI and Age are second and third most important.

**Q32. What is overfitting? How did you prevent it?**
> Overfitting is when a model learns training data too well including noise, failing on new data. We prevented it by:
> - Using 5-fold cross-validation
> - Limiting Decision Tree depth to 6
> - Using Random Forest (ensemble reduces variance)
> - Using 80/20 stratified train-test split

---

## SECTION 9: GENERAL ML THEORY

**Q33. What is supervised vs unsupervised learning?**
> - **Supervised**: Labeled data — model learns input→output mapping. Both our diseases use supervised (Classification).
> - **Unsupervised**: No labels — finds hidden patterns. PCA is unsupervised.

**Q34. What is classification vs regression?**
> - **Classification**: Predicts a discrete category (Diabetic/Not Diabetic)
> - **Regression**: Predicts a continuous value (e.g., exact glucose level)
> Our project uses classification.

**Q35. What is the bias-variance tradeoff?**
> - **High Bias** (Underfitting): Model too simple, misses patterns (e.g., Linear Regression on non-linear data)
> - **High Variance** (Overfitting): Model too complex, memorizes noise
> - **Goal**: Balance both. Random Forest achieves this by averaging many trees.

---

## SECTION 10: HOW TO EXPLAIN TO TEACHER

**Opening statement:**
> "Ma'am/Sir, our project is a Dual Disease Prediction System. We built a complete ML pipeline that predicts two diseases — Diabetes and Breast Cancer — using 6 different classification algorithms. We compared all models, selected the best one for each disease, and deployed it as a real-time Streamlit web application."

**Demo flow:**
1. Show the web app → Home Page → explain the concept
2. Enter diabetes values → show prediction result
3. Enter cancer values → show prediction result
4. Show Analytics page → model comparison bars + feature importance
5. Show Colab notebook → EDA graphs, heatmap, confusion matrix, ROC curves
6. Explain novelty vs regular single-disease projects

**Closing:**
> "The key innovation is that most projects predict one disease, but ours creates a unified platform for two diseases, with automatic model selection, visual analytics, and a user-friendly web interface that a non-technical doctor or nurse can use directly."
