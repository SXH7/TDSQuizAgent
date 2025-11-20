import os
import google.generativeai as genai

try:
    api_key = os.environ["GOOGLE_API_KEY"]
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable is not set.")
    else:
        genai.configure(api_key=api_key)
        print("Available models that support 'generateContent':")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
except Exception as e:
    print(f"An error occurred: {e}")
