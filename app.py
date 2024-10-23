from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from transformers import pipeline
import os
import PyPDF2

app = Flask(__name__)

# Set the folder to store uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Explicitly specify the model name and revision for Hugging Face's summarizer pipeline
model_name = "sshleifer/distilbart-cnn-12-6"  # You can switch to a smaller model like distilbart-cnn-6-6 if needed
revision = "a4f8f3e"  # Ensure stable model version by specifying the revision

# Initialize the Huggingface summarizer pipeline with explicit model and revision
summarizer = pipeline("summarization", model=model_name, revision=revision)

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Helper function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"
    return text

# Helper function to summarize the text
def summarize_text(text):
    if not text or len(text.strip()) == 0:
        return "No text found in the PDF."

    input_length = len(text.split())  # Get the word count of the input text
    # Define min and max length based on input text length
    max_length = min(150, max(25, int(input_length * 0.2)))  # Ensure max_length isn't too large
    min_length = max(20, int(max_length * 0.5))  # Ensure min_length isn't too small

    # Log to check if lengths are set appropriately
    print(f"Input length: {input_length}, Max length: {max_length}, Min length: {min_length}")

    if input_length <= 50:  # If input is very short, handle it gracefully
        max_length = 25  # Set a very short max_length for small inputs
        min_length = 10  # Set a minimal length for such cases
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']

    if len(text) > 1000:  # For long texts, split into chunks
        summary = []
        for i in range(0, len(text), 1000):
            chunk = text[i:i + 1000]
            summary_chunk = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            summary.append(summary_chunk[0]['summary_text'])
        return ' '.join(summary)
    else:  # For shorter text inputs
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle PDF upload and summarization
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Extract text from the uploaded PDF
        extracted_text = extract_text_from_pdf(file_path)

        # Summarize the extracted text
        summary = summarize_text(extracted_text)

        return render_template('result.html', summary=summary)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
