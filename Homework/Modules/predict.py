import pickle


# Configuration
MODEL_FILE = 'model_C=1.0.bin'


def load_model(model_file):
    """Load the trained model and vectorizer."""
    with open(model_file, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    print(f"Model loaded from {model_file}")
    return dv, model


def predict_single(customer, dv, model):
    """Predict churn probability for a single customer."""
    X = dv.transform([customer])
    churn_probability = model.predict_proba(X)[0, 1]
    return churn_probability


def predict_batch(customers, dv, model):
    """Predict churn probability for multiple customers."""
    X = dv.transform(customers)
    churn_probabilities = model.predict_proba(X)[:, 1]
    return churn_probabilities


def main():
    # Load the trained model
    dv, model = load_model(MODEL_FILE)
    
    # Example customer
    customer = {
        'gender': 'female',
        'seniorcitizen': 0,
        'partner': 'yes',
        'dependents': 'no',
        'phoneservice': 'no',
        'multiplelines': 'no_phone_service',
        'internetservice': 'dsl',
        'onlinesecurity': 'no',
        'onlinebackup': 'yes',
        'deviceprotection': 'no',
        'techsupport': 'no',
        'streamingtv': 'no',
        'streamingmovies': 'no',
        'contract': 'month-to-month',
        'paperlessbilling': 'yes',
        'paymentmethod': 'electronic_check',
        'tenure': 1,
        'monthlycharges': 29.85,
        'totalcharges': 29.85
    }
    
    # Make prediction
    churn_prob = predict_single(customer, dv, model)
    
    print(f"\nCustomer Details:")
    print(f"  - Contract: {customer['contract']}")
    print(f"  - Tenure: {customer['tenure']} months")
    print(f"  - Monthly Charges: ${customer['monthlycharges']}")
    print(f"\nChurn Probability: {churn_prob:.3f}")
    print(f"Churn Risk: {'HIGH' if churn_prob > 0.5 else 'LOW'}")

if __name__ == "__main__":
    main()