from waitress import serve
from predict_test import app

if __name__ == '__main__':
    print("Production server starting on port 9696...")
    serve(app, host='0.0.0.0', port=9696, threads=4)