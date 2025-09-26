📌 Task 4 – Classification with Logistic Regression (Breast Cancer Dataset)
🔹 Objective

The goal of this task is to build a binary classifier using Logistic Regression to predict whether a breast tumor is malignant (0) or benign (1).

🔹 Tools Used

Python

Pandas → Data handling

Matplotlib → Visualization (ROC curve)

Scikit-learn → Logistic Regression, metrics, preprocessing

🔹 Steps Performed

-Choose Dataset

Used the Breast Cancer Wisconsin dataset from sklearn.datasets.

Features: 30 numeric tumor characteristics.

Target: 0 = malignant, 1 = benign.

-Preprocessing

Split dataset into train (80%) and test (20%) sets.

Standardized features using StandardScaler (important for Logistic Regression).

-Model Training

Built a Logistic Regression model using sklearn.linear_model.LogisticRegression.

Trained on training data.

-Model Evaluation

Confusion Matrix → shows True Positives, False Positives, etc.

Classification Report → Precision, Recall, F1-score, Accuracy.

ROC-AUC Score → measured the ability of the model to distinguish classes.

ROC Curve plotted to visualize trade-off between TPR and FPR.

-Threshold Tuning & Sigmoid Function

Default threshold = 0.5.

Experimented with changing threshold (e.g., 0.6) to balance sensitivity vs specificity.

Explained the Sigmoid function:

𝜎(𝑧)=1/1+e^-z

Converts linear output into probability (0–1).

🔹 Results

Logistic Regression achieved high accuracy on breast cancer classification.

Confusion Matrix showed very few misclassifications.

Precision & Recall were strong, making it reliable for medical prediction.

ROC-AUC Score close to 1, showing excellent separability of classes.

🔹 Learning Outcome

Understood the concept of Logistic Regression as a classifier.

Learned how to standardize data, train, and evaluate classification models.

Gained hands-on practice with confusion matrix, precision, recall, ROC-AUC.

Learned how sigmoid function maps predictions into probabilities.

Understood how threshold tuning affects model predictions.
