from translator import translator_instance
import os
import sys

def verify():
    print("--- Verifying Environment ---")
    token = os.getenv("HF_TOKEN")
    repo = os.getenv("HF_REPO_ID")
    
    if token:
        print(f"✅ HF_TOKEN found: {token[:4]}...")
    else:
        print("❌ HF_TOKEN NOT found")
        
    if repo:
        print(f"✅ HF_REPO_ID found: {repo}")
    else:
        print("❌ HF_REPO_ID NOT found")

    print("\n--- Verifying Translation ---")
    try:
        # Test distinct word to trigger API (not in local corpus)
        text = "Hello world"
        result = translator_instance.translate(text)
        print(f"Translation result for '{text}': {result}")
        
        if result and "Error" not in result:
            print("✅ Translation API Call Successful")
        else:
            print("❌ Translation API Call Failed")
            
    except Exception as e:
        print(f"❌ Exception during translation: {e}")

if __name__ == "__main__":
    verify()
