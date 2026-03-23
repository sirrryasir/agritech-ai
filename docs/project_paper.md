# AgriTech AI: Machine Learning for Crop Recommendation
**Final Project Paper - DS & ML Bootcamp**

## 1. Problem Statement & Motivation
Agriculture plays a critical role in the global economy, particularly in developing nations like Somalia. However, many farmers still rely on traditional knowledge and intuition to decide which crops to plant. This can lead to suboptimal yields, especially in the face of changing climate conditions.

The motivation behind "AgriTech AI" is to empower farmers with data-driven decision-making. By analyzing soil nutrients (Nitrogen, Phosphorus, Potassium), environmental metrics (Temperature, Humidity, Rainfall), and the soil's pH level, this Machine Learning system predicts the absolute best crop to cultivate. This maximizes productivity, reduces agricultural waste, and ensures sustainable farming practices.

## 2. Dataset & Preprocessing
The dataset used for this project is the **Crop Recommendation Dataset** sourced from Kaggle.
- **Size:** 2,200 samples. This satisfies the bootcamp's requirement of at least 1,000 samples.
- **Features:** 7 numerical features (`N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`).
- **Target:** 1 categorical target (`label`), which contains 22 unique crop names (e.g., rice, maize, apple, orange).

### Preprocessing Steps:
1. **Label Encoding:** Since ML algorithms require numerical target variables, `LabelEncoder` from `scikit-learn` was used to transform the 22 crop names into integers ranging from 0 to 21.
2. **Train-Test Split:** The dataset was partitioned into an 80% training set and a 20% testing set using stratified splitting to ensure each crop was equally represented in both sets.
3. **Feature Scaling:** A `StandardScaler` was applied to normalize the input features. This is particularly crucial for algorithms that calculate distances, ensuring features like rainfall (which can be >200mm) do not artificially outweigh pH (which ranges from 0-14).

\*The fitted `StandardScaler` and `LabelEncoder` were subsequently exported using `joblib` so that new user inputs from the API can be accurately processed.\*

## 3. Algorithms & Methodology
We selected a Multi-Class Classification approach to solve this problem. Two distinctly different algorithms were trained and compared:

### A. Gaussian Naive Bayes (GNB)
Naive Bayes was selected due to its simplicity, speed, and effectiveness as a baseline model. Since our dataset consists of continuous numerical features, the Gaussian variant of Naive Bayes was utilized, which assumes that the continuous values associated with each class are distributed according to a normal (Gaussian) distribution.

### B. Random Forest Classifier
Random Forest is a powerful ensemble method that constructs a multitude of decision trees during training and outputs the mode of the classes for prediction. It was chosen because:
- It inherently handles multi-class classification very well.
- It is highly resistant to overfitting compared to a single Decision Tree.
- It can capture complex, non-linear relationships between soil metrics and crop viability.

## 4. Results & Discussion
Both models were evaluated on the 20% hold-out test set using Accuracy and the Classification Report (Precision, Recall, F1-Score).

- **Gaussian Naive Bayes Accuracy:** ~99.5%
- **Random Forest Accuracy:** ~99.5% to 100%

Both algorithms showed exceptional performance, primarily because the dataset contains distinct, well-separated clusters for each crop's optimal growing conditions.

**Conclusion:** We selected the **Random Forest Classifier** as our final production model because ensemble methods generalize slightly better to unseen, real-world edge cases.

### Sanity Checks
We performed manual sanity checks bypassing specific arrays to the model. For instance, high rainfall (>200mm), high humidity (>80%), and high NPK levels correctly predicted **Rice**, mirroring real-world agricultural requirements.

## 5. Deployment Architecture
To make the model accessible to end-users (farmers/agronomists), it was deployed using a two-tier architecture:

### The Backend (Flask API)
A RESTful API was developed using Python's Flask framework.
- At startup, the API loads the `scaler`, `label_encoder`, and `best_model` from the `models/` directory.
- It exposes a `POST /predict` endpoint.
- When a user submits JSON data containing soil metrics, the API dynamically applies the exact same standard scaling used during training, executes `model.predict()`, and uses `inverse_transform` to return the human-readable crop string.

### The Frontend (Next.js)
A modern, highly interactive dashboard was built using **Next.js** and **Tailwind CSS**. The dashboard provides a sleek UI where users can input parameters via a form, successfully hiding the complexity of JSON formulation and cURL requests. It fetches data from the Flask API and prominently displays the recommended crop.

## 6. Lessons Learned
Completing this capstone project provided several key takeaways:
1. **The Importance of the ML Pipeline:** Saving not just the final model, but also the Scalers and Encoders, is essential for deployment. If new input data isn't scaled exactly like the training data, the model's predictions will be completely inaccurate.
2. **Cross-Origin Resource Sharing (CORS):** Connecting a React frontend (port 3000) to a Flask backend (port 5000) introduced CORS errors, teaching the practical necessity of utilizing `flask-cors` in web development.
3. **Model Selection:** While advanced deep learning models receive lots of hype, well-tuned classical Machine Learning models like Random Forest are incredibly powerful and computationally efficient for tabular data tasks.

*This project successfully achieved its goal of building an end-to-end Machine Learning ecosystem, solidifying the concepts learned throughout the DS & ML Bootcamp.*
