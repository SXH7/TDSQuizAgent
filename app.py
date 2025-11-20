from flask import Flask, request, jsonify
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import requests
import json
import re
from urllib.parse import urljoin
import os
import google.generativeai as genai
from dotenv import load_dotenv
import whisper
import tempfile
import pandas as pd
from io import StringIO

load_dotenv()

app = Flask(__name__)

# IMPORTANT: Load your actual secret and email from environment variables
YOUR_SECRET = os.environ.get("QUIZ_SECRET")
YOUR_EMAIL = os.environ.get("QUIZ_EMAIL")

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def execute_python_code(code: str, csv_data: str = None) -> str:
    """
    Executes Python code and returns its output.
    If csv_data is provided, it's made available as a pandas DataFrame 'df'.
    """
    local_vars = {}
    if csv_data:
        local_vars['pd'] = pd
        local_vars['StringIO'] = StringIO
        local_vars['df'] = pd.read_csv(StringIO(csv_data), header=None)
    
    try:
        # Use exec to run the code, capturing stdout
        output_capture = StringIO()
        # Redirect stdout to our StringIO object
        import sys
        sys.stdout = output_capture
        
        exec(code, {}, local_vars)
        
        sys.stdout = sys.__stdout__ # Restore stdout
        
        result = output_capture.getvalue().strip()
        if not result and 'df' in local_vars: # If no explicit print, try to get the last expression's value
            # This is a heuristic: try to get the last line's evaluation if it's an expression
            last_line = code.strip().split('\n')[-1]
            try:
                result = str(eval(last_line, {}, local_vars))
            except Exception:
                pass # Not an evaluable expression
        return result
    except Exception as e:
        return f"Error executing code: {e}"

def get_next_action_from_llm(content: str, quiz_url: str, context: dict) -> dict:
    """
    Uses the Gemini LLM to determine the next action to take.
    """
    model = genai.GenerativeModel('models/gemini-flash-latest')
    
    context_str = "\n".join([f"--- {key.upper()} ---\n{value}" for key, value in context.items()])
    prompt = f"""
You are an expert quiz-solving agent driving a web browser.
Your goal is to solve the quiz with perfect reliability and NO HALLUCINATIONS.

You will be given:
- The HTML content of the current page
- A context dictionary with previously gathered data

==============================
STRICT NON-HALLUCINATION RULES
==============================

1. **You may ONLY choose a SCRAPE action if the URL you scrape appears
   *literally and exactly* inside the HTML Content.**
   - Do NOT invent filenames like "/data.csv", "file.csv", or any URL that
     does not appear verbatim in the HTML Content string.
   - If no links appear, DO NOT SCRAPE.

2. **If the HTML does NOT contain any .csv, .json, .pdf, or download links,
   you MUST assume no data file exists.**
   - Do NOT assume the page “probably” has a CSV.
   - Do NOT assume the quiz “usually” needs a file.

3. **If you are uncertain whether to SCRAPE or ANSWER, ALWAYS choose ANSWER.**

4. **You are NOT allowed to infer or guess the existence of a data file.**
   - If the HTML content does not reference a file, no file exists.

5. **The only URLs allowed for SCRAPE are:**
   - URLs explicitly visible in the HTML as href=, src=, or text content
   - Absolute or relative URLs literally present in the HTML

6. **Do NOT SCRAPE submission URLs** (those containing “submit”, “answer”, “post”).

=========================
ACTION DEFINITIONS
=========================

1. SCRAPE action:
   Use only when the HTML explicitly contains a valid data URL (e.g., a real CSV link).
   Respond like:
     {{"action": "SCRAPE", "url": "<exact_url_copied_from_HTML>"}}

2. ANSWER action:
   Use this when:
   - You already have all required data
   - OR no data file exists
   - OR you are unsure
   - OR the page has instructions but no downloadable file

If CSV data exists in context, and analysis is needed, produce Python code.
Your Python code will be executed with a pandas DataFrame named `df`.

Your ANSWER JSON should look like:
   {{"action": "ANSWER", "answer": "<final_answer>"}}
or, if python analysis is needed:
   {{"action": "ANSWER", "answer": "print(...)"}}

=========================
STATE INFORMATION
=========================

Page URL: {quiz_url}
HTML Content:
---
{content}
---
Gathered Context:
{context_str}

=========================
DECISION PROCESS
=========================
1. First, check if instructions exist in the context (e.g., transcribed audio).
2. If the instructions refer to a file:
   - Verify that the HTML actually contains a link to that file.
   - If no such link exists → DO NOT SCRAPE. Assume no file exists.
3. If csv_data is present, inspect columns before analysis.
4. If the HTML contains a real data link, you may SCRAPE it.
5. Otherwise, default to ANSWER.

Your output MUST be valid JSON with no extra text.

Now decide the next action.
"""
    response = model.generate_content(prompt)
    try:
        # The response text might be wrapped in ```json ... ```, so we need to extract it.
        match = re.search(r"```json\n(.*?)\n```", response.text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = response.text
        return json.loads(json_str.strip())
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error decoding LLM response: {e}")
        print(f"Raw response: {response.text}")
        return {"action": "ANSWER", "answer": "Error - could not parse LLM response"}

def _solve_single_quiz(quiz_url: str, email: str, secret: str, context: dict = None):
    """
    Solves a single quiz task given its URL by acting as an agent.
    Returns the submission response from the quiz server.
    """
    print(f"Starting to solve quiz at: {quiz_url}")
    
    answer = None
    current_url_to_process = quiz_url
    context = {} # To hold transcribed audio, csv data, etc.

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # Perception loop: Agent gathers information until it's ready to answer.
        while True: 
            print(f"Processing URL: {current_url_to_process}")
            page.goto(current_url_to_process)
            content = page.content()
            soup = BeautifulSoup(content, 'html.parser')

            # --- Perception Step: Look for Audio ---
            if "transcribed_audio" not in context:
                audio_url = None
                # Look for <audio src="...">
                audio_tag = soup.find('audio')
                if audio_tag and audio_tag.get('src'):
                    audio_url = urljoin(current_url_to_process, audio_tag['src'])
                
                # Fallback: Look for <a> tags linking to audio files
                if not audio_url:
                    audio_links = soup.find_all('a', href=re.compile(r'\.(mp3|wav|ogg|opus)$'))
                    if audio_links:
                        audio_url = urljoin(current_url_to_process, audio_links[0]['href'])

                if audio_url:
                    print(f"Found audio file: {audio_url}")
                    
                    audio_response = requests.get(audio_url)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".opus") as tmp_audio: # Use .opus suffix for consistency
                        tmp_audio.write(audio_response.content)
                        tmp_audio_path = tmp_audio.name
                    
                    print("Transcribing audio... (This may take a moment)")
                    model = whisper.load_model("tiny") # Use a small model for speed
                    result = model.transcribe(tmp_audio_path)
                    transcribed_text = result["text"]
                    os.remove(tmp_audio_path)

                    print(f"Transcription complete: {transcribed_text}")
                    context["transcribed_audio"] = transcribed_text
                else:
                    print("No audio files found on the page.")

            # --- Perception Step: Look for Cutoff Value ---
            # Removed hardcoded cutoff value extraction. LLM will extract from content.

            # --- Action Step: Ask LLM what to do next ---
            action_json = get_next_action_from_llm(content, current_url_to_process, context)
            action = action_json.get("action")

            if action == "SCRAPE":
                scrape_url = action_json.get("url")
                if not scrape_url:
                    answer = "Error: LLM chose SCRAPE but provided no URL."
                    break
                
                absolute_scrape_url = urljoin(current_url_to_process, scrape_url)
                print(f"LLM chose to SCRAPE: {absolute_scrape_url}")

                if absolute_scrape_url.endswith('.csv'):
                    print("Detected CSV file. Downloading content...")
                    csv_content = requests.get(absolute_scrape_url).text
                    context["csv_data"] = csv_content
                    print("CSV content added to context. Re-evaluating...")
                    # Stay on the same page, the loop will re-prompt the LLM with the new context
                    current_url_to_process = quiz_url 
                else:
                    # It's a web page to scrape
                    current_url_to_process = absolute_scrape_url
                
                continue # Continue the perception loop

            elif action == "ANSWER":
                answer_content = action_json.get("answer")
                print(f"LLM chose to ANSWER: {answer_content}")

                # Check if the answer is Python code
                if "pd." in answer_content or "df[" in answer_content:
                    print("Executing LLM-generated Python code...")
                    execution_result = execute_python_code(answer_content, context.get("csv_data"))
                    print(f"Code execution result: {execution_result}")
                    answer = execution_result
                else:
                    answer = answer_content
                break # Exit the loop to submit the answer
            
            else:
                answer = f"Error: Unknown action from LLM: {action}"
                break

        # --- Submission Step ---
        print(f"Finding submission URL on original page: {quiz_url}")
        page.goto(quiz_url)
        content = page.content()
        soup = BeautifulSoup(content, 'html.parser')
        submission_url = None

        submit_link = soup.find('a', href=re.compile("submit"))
        if submit_link:
            submission_url = urljoin(quiz_url, submit_link['href'])
        else:
            submit_origin_span = soup.find('span', class_='origin')
            if submit_origin_span:
                base_url = submit_origin_span.text.strip()
                submit_path = submit_origin_span.next_sibling.strip()
                submission_url = base_url + submit_path
            else:
                body_text = soup.body.get_text(separator=' ')
                match = re.search(r"POST this JSON to (https?://[^ ]+)", body_text)
                if match:
                    submission_url = match.group(1).strip()

        if not submission_url:
            print("Error: Could not find submission URL in quiz page.")
            return {"correct": False, "reason": "Could not find submission URL"}

    # Convert to int if the answer is a number string
    if isinstance(answer, str) and answer.isdigit():
        answer = int(answer)

    # Construct the submission payload
    submission_payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer
    }

    print(f"Submitting answer to: {submission_url}")
    print(f"Payload: {json.dumps(submission_payload, indent=2)}")

    try:
        response = requests.post(submission_url, json=submission_payload)
        response.raise_for_status() # Raise an exception for HTTP errors
        submission_response = response.json()
        print(f"Submission successful! Response: {submission_response}")
        return submission_response
    except requests.exceptions.RequestException as e:
        print(f"Submission failed: {e}")
        return {"correct": False, "reason": f"Submission failed: {e}"}

@app.route('/', methods=['POST'])
def solve_quiz():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    email = data.get('email')
    secret = data.get('secret')
    initial_url = data.get('url')

    if secret != YOUR_SECRET:
        return jsonify({"error": "Invalid secret"}), 403
    
    if email != YOUR_EMAIL:
        return jsonify({"error": "Invalid email"}), 403

    current_url = initial_url
    all_submission_responses = []
    
    while current_url:
        submission_result = _solve_single_quiz(current_url, email, secret)
        all_submission_responses.append(submission_result)

        if submission_result.get('correct') and submission_result.get('url'):
            current_url = submission_result['url']
        else:
            current_url = None # Stop the loop

    final_status = "completed" if all_submission_responses and all_submission_responses[-1].get('correct') else "failed"
    return jsonify({
        "message": f"Quiz sequence {final_status}",
        "initial_url": initial_url,
        "final_status": final_status,
        "submission_history": all_submission_responses
    }), 200

if __name__ == '__main__':
    app.run(debug=True, port=5001)