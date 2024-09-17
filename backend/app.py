from flask import Flask, request, jsonify, session
import torch
from PIL import Image
from torchvision import transforms
import os
import torch.nn as nn
import torchvision.models as models
from groq import Groq
from dotenv import load_dotenv
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load environment variables from the .env file
load_dotenv()

# Define your image classification model
class DiseaseSpecificModel(nn.Module):
    def __init__(self, num_classes=8):
        super(DiseaseSpecificModel, self).__init__()
        self.optic_disc_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.macula_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Modify the output layers
        self.optic_disc_model.fc = nn.Linear(self.optic_disc_model.fc.in_features, 512)
        self.macula_model.fc = nn.Linear(self.macula_model.fc.in_features, 512)

        # Add dropout before the final classification layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 2, num_classes)
        )

    def forward(self, left_img, right_img):
        left_features = self.optic_disc_model(left_img)
        right_features = self.macula_model(right_img)
        
        combined_features = torch.cat((left_features, right_features), dim=1)
        output = self.classifier(combined_features)
        return output

# Load and configure the image classification model
model_path = 'best_model.pth'
disease_model = DiseaseSpecificModel(num_classes=8)
disease_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
disease_model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define class-to-disease mapping and treatment mapping
class_to_disease = {
    0: 'Normal',
    1: 'Diabetes',
    2: 'Glaucoma',
    3: 'Cataract',
    4: 'Age-related Macular Degeneration (AMD)',
    5: 'Hypertension',
    6: 'Myopia',
    7: 'Other Diseases/Abnormalities'
}

disease_treatment = {
    'Normal': 'No treatment required. Maintain regular check-ups.',
    'Diabetes': 'Manage blood sugar levels, regular eye exams, possible laser treatment or surgery.',
    'Glaucoma': 'Medications, laser treatment, or surgery to lower intraocular pressure.',
    'Cataract': 'Surgery to replace the cloudy lens with an artificial lens.',
    'Age-related Macular Degeneration (AMD)': 'Anti-VEGF injections, laser therapy, lifestyle changes (diet, smoking cessation).',
    'Hypertension': 'Blood pressure control, lifestyle changes, medications.',
    'Myopia': 'Corrective lenses, orthokeratology, or refractive surgery for severe cases.',
    'Other Diseases/Abnormalities': 'Consult with an ophthalmologist for specific treatment options.'
}

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Function to get the LLM response using Groq
def get_llm_response(disease, treatment, user_prompt=None):
    # Generate prompt
    if user_prompt:
        prompt = f"{user_prompt}. Provide a factual answer from a medical perspective."
    else:
        if disease == "Normal":
            prompt = "The patient shows no signs of disease. Provide guidelines for maintaining optimal eye health, including lifestyle recommendations."
        else:
            prompt = f"A patient is diagnosed with {disease}. Provide a detailed medical report that includes recommended treatment options, lifestyle changes, possible outcomes, and future prognosis. Format your answer in a easy to look, digestable way."

    # system_message = "You are a concise medical assistant. Provide brief, direct answers."
        # Updated system message reflecting formatting instructions
    system_message = (
        "You are a medical assistant. Provide brief, direct answers broken down into easy-to-read paragraphs. "
        "To format your responses: "
        "- Use double asterisks ** before and after any word or phrase to make it **bold**. "
        "- Use single asterisks * at the beginning and end of a phrase to format it as a bullet point. "
        "- Separate each paragraph with two newlines (\\n\\n) for clarity. "
        "Your goal is to ensure that the information is clear, well-structured, and easy to digest."
    )
    print(f"Generated Prompt: {prompt}")

    # Generate text using Groq API
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192"
        )
        response_text = response.choices[0].message.content
        print(f"LLM Response: {response_text}")
    except Exception as e:
        print(f"Error during text generation: {e}")
        return "Error during text generation."

    return response_text

# API endpoint for handling user prompts
@app.route('/api/chat', methods=['POST'])
def handle_chat():
    if request.json and 'prompt' in request.json:
        user_prompt = request.json['prompt']
        session['chat_history'] = session.get('chat_history', [])
        session['chat_history'].append({'role': 'user', 'content': user_prompt})

        llm_response = get_llm_response(None, None, user_prompt)
        session['chat_history'].append({'role': 'bot', 'content': llm_response})

        return jsonify({"response": llm_response, "chat_history": session['chat_history']})
    return jsonify({"error": "Invalid input"}), 400

# API endpoint for handling file upload
@app.route('/api/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Open and preprocess the image
    try:
        image = Image.open(filepath).convert('RGB')
        image = transform(image).unsqueeze(0)
    except Exception as e:
        return jsonify({"error": f"Error processing image: {e}"}), 400

    with torch.no_grad():
        left_img = image
        right_img = image
        output = disease_model(left_img, right_img)
        predicted_idx = torch.argmax(output, dim=1).item()
        predicted_disease = class_to_disease[predicted_idx]

    treatment = disease_treatment.get(predicted_disease, 'No treatment available')
    llm_response = get_llm_response(predicted_disease, treatment)

    diagnosis = f'**Diagnosed**: {predicted_disease}. \n\n **Suggested Treatment**: {treatment}. \n\n {llm_response}'
    session['chat_history'] = session.get('chat_history', [])
    session['chat_history'].append({'role': 'bot', 'content': diagnosis})

    return jsonify({"diagnosis": diagnosis, "chat_history": session['chat_history']})

if __name__ == '__main__':
    app.run(debug=True)
