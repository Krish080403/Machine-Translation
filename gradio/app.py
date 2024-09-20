from flask import Flask, request, render_template_string
import os
from translate import translate_text

app = Flask(__name__)

html_template = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <title>Hindi to Mundari Translator</title>
    <style>
      body {
        padding: 20px;
        background-color: #121212;
        color: #e0e0e0;
        font-family: 'Arial', sans-serif;
      }
      .container {
        max-width: 800px;
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
      }
      .container:hover {
        transform: scale(1.02);
      }
      textarea {
        resize: none;
        background-color: #2e2e2e;
        color: #e0e0e0;
        border: 1px solid #444;
        border-radius: 4px;
      }
      button {
        transition: background-color 0.3s ease;
      }
      button:hover {
        background-color: #0056b3;
      }
      .spinner-border {
        width: 3rem;
        height: 3rem;
        border-width: 0.4em;
        animation: spin 1s linear infinite;
      }
      @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
      .btn-primary {
        background-color: #007bff;
        border: none;
      }
      .btn-primary:focus, .btn-primary:active {
        box-shadow: none;
      }
      .text-center {
        margin-top: 20px;
      }
      .error-message {
        color: #ff4d4d;
        font-weight: bold;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <h1 class="mt-5 mb-4 text-center"> Hindi to Mundari Translator </h1>
      <form method="post" action="/">
        <div class="form-group">
          <label for="inputText">Enter up to 10 sentences in Hindi:</label>
          <textarea class="form-control" id="inputText" name="inputText" rows="10" required>{{ input_text }}</textarea>
        </div>
        <button type="submit" class="btn btn-primary btn-block">Translate</button>
      </form>
      {% if processing %}
        <div class="text-center mt-5">
          <div class="spinner-border" role="status">
            <span class="sr-only">Processing...</span>
          </div>
          <p class="mt-3">Processing... Please wait.</p>
        </div>
      {% elif translated_text %}
        <h2 class="mt-5">Translated text in Mundari :</h2>
        <textarea class="form-control" rows="10" readonly>{{ translated_text }}</textarea>
      {% elif error_message %}
        <p class="error-message">{{ error_message }}</p>
      {% endif %}
    </div>
  </body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def translate():
    processing = False
    translated_text = None
    error_message = None
    input_text = ""  # Default to empty

    if request.method == "POST":
        processing = True
        input_text = request.form["inputText"]
        try:
            translated_text = translate_text(input_text)
        except Exception as e:
            error_message = f"An error occurred: {e}"

        processing = False

    return render_template_string(html_template, translated_text=translated_text, processing=processing, error_message=error_message, input_text=input_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001,debug=True)
