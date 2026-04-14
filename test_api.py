import requests
import json
import os

# Test 1: Valid threshold 0.5
test_img_path = 'dataset/test/real'
test_files = os.listdir(test_img_path)

if test_files:
    img_file = os.path.join(test_img_path, test_files[0])
    
    # Test with valid threshold 0.5
    print("=" * 60)
    print("Test 1 - Valid threshold (0.5):")
    print("=" * 60)
    with open(img_file, 'rb') as f:
        files = {'file': f}
        response = requests.post('http://127.0.0.1:8000/predict?threshold=0.5', files=files)
        print(f'Status Code: {response.status_code}')
        if response.status_code == 200:
            result = response.json()
            print(f'✅ Valid threshold accepted')
            print(f'Threshold Used: {result["threshold_used"]}')
            print(f'Prediction: {result["prediction"]}')
            print(f'Combined Fraud Score: {result["combined_fraud_score"]:.4f}')
        else:
            print(f'❌ Error: {response.json()}')
    
    # Test with invalid threshold 0.8
    print("\n" + "=" * 60)
    print("Test 2 - Invalid threshold (0.8):")
    print("=" * 60)
    with open(img_file, 'rb') as f:
        files = {'file': f}
        response = requests.post('http://127.0.0.1:8000/predict?threshold=0.8', files=files)
        print(f'Status Code: {response.status_code}')
        if response.status_code == 400:
            print(f'✅ Invalid threshold rejected correctly')
            print(f'Error Message: {response.json()["detail"]}')
        else:
            print(f'Unexpected response: {response.json()}')
    
    # Test with default threshold
    print("\n" + "=" * 60)
    print("Test 3 - Default threshold (0.65):")
    print("=" * 60)
    with open(img_file, 'rb') as f:
        files = {'file': f}
        response = requests.post('http://127.0.0.1:8000/predict', files=files)
        print(f'Status Code: {response.status_code}')
        if response.status_code == 200:
            result = response.json()
            print(f'✅ Request successful')
            print(f'Threshold Used: {result["threshold_used"]}')
            print(f'Prediction: {result["prediction"]}')
            print(f'Combined Fraud Score: {result["combined_fraud_score"]:.4f}')
        else:
            print(f'❌ Error: {response.json()}')
    
    print("\n" + "=" * 60)
    print("API Testing Complete! ✅")
    print("=" * 60)
else:
    print("❌ No test images found in dataset/test/real/")
