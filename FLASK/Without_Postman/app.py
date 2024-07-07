from flask import Flask,request,jsonify
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import joblib

app = Flask(__name__)

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
     import os

     # Define the file name
     model_file = 'File/iris_model.pkl'

     # Check if the file exists
     if os.path.exists(model_file):
          print(f"The file '{model_file}' exists.")
          clf = joblib.load(model_file)
     else:
          print(f"The file '{model_file}' does not exist.")
          # Load the Iris dataset
          iris = load_iris()
          X = iris.data
          y = iris.target
          feature_names = iris.feature_names
          target_names = iris.target_names

          # Train a decision tree classifier
          clf = DecisionTreeClassifier()
          clf.fit(X, y)
          print('Training done')
          # Save the model
          joblib.dump(clf, 'File/iris_model.pkl')
     try:
        # Get input values from form
        input_features = [float(x) for x in request.form.values()]
        print(input_features)
        # Make prediction
        prediction = clf.predict([input_features])
        print(prediction)
        predicted_species = target_names[prediction]
        print(predicted_species)
        return jsonify({'predicted_species':predicted_species[0]})
     except:
        return jsonify({'message':'Error: Please enter valid input values.'})

if __name__ == '__main__':
    app.run(debug=True)
