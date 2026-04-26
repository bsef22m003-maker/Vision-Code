"""
Simple test script to understand FastAPI usage
Run this AFTER starting the API server with:
uvicorn api_server:app --reload --port 8000
"""

import requests
import time

# API base URL
API_URL = "http://localhost:8000"

print("=" * 60)
print("EXAM PROCTORING API - SIMPLE TEST")
print("=" * 60)

# Step 1: Check if API is online
print("\n1️⃣ Checking API health...")
try:
    response = requests.get(f"{API_URL}/")
    print(f"   Status: {response.json()['status']}")
    print(f"   Response: {response.json()}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    print(f"   Make sure the server is running!")
    print(f"   Run: uvicorn api_server:app --reload --port 8000")
    exit(1)

# Step 2: Create a session
print("\n2️⃣ Creating a new session...")
session_data = {
    "user_id": "student123",
    "exam_id": "math_final_2024",
    "use_arcface": False,
    "sound_threshold": 0.01
}
response = requests.post(f"{API_URL}/api/sessions/create", json=session_data)
result = response.json()
session_id = result['session_id']
print(f"   ✅ Session created!")
print(f"   Session ID: {session_id}")

# Step 3: Get session status
print("\n3️⃣ Checking session status...")
response = requests.get(f"{API_URL}/api/sessions/{session_id}/status")
status = response.json()
print(f"   User ID: {status['user_id']}")
print(f"   Exam ID: {status['exam_id']}")
print(f"   Status: {status['status']}")
print(f"   Face Registered: {status['face_registered']}")
print(f"   Calibrated: {status['calibrated']}")

# Step 4: List all sessions
print("\n4️⃣ Listing all active sessions...")
response = requests.get(f"{API_URL}/api/sessions")
sessions = response.json()
print(f"   Active Sessions: {sessions['active_sessions']}")

# Step 5: End the session
print("\n5️⃣ Ending session...")
response = requests.delete(f"{API_URL}/api/sessions/{session_id}")
result = response.json()
print(f"   ✅ Session ended successfully!")

print("\n" + "=" * 60)
print("TEST COMPLETED SUCCESSFULLY! 🎉")
print("=" * 60)
print("\nNext steps:")
print("1. Try the interactive docs: http://localhost:8000/docs")
print("2. Run the complete workflow: python api_client_example.py")
print("3. Test with camera: python test_complete_workflow.py")