from flask import Flask, request, jsonify, session, make_response, url_for
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
from flask_caching import Cache
from hashlib import md5
from flask_session import Session
from redis import Redis

# Load environment variables from the .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = os.getenv('FLASK_SECRET_KEY')
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Initialize the cache
cache = Cache(config={'CACHE_TYPE': 'simple'})  # Simple in-memory cache
cache.init_app(app)

# Configure session to use Redis
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_PERMANENT'] = False  # Set to True if you want sessions to persist between server restarts
app.config['SESSION_USE_SIGNER'] = True  # Sign the session to prevent tampering
app.config['SESSION_KEY_PREFIX'] = 'your_app_session:'  # Prefix for Redis keys
app.config['SESSION_REDIS'] = Redis(host='localhost', port=6379, db=0)

from flask_cors import CORS

CORS(app, supports_credentials=True, resources={r"/*": {"origins": "http://localhost:3000"}})

# Initialize the session extension
Session(app)

def is_health_related(prompt):
    health_keywords = [
        'health', 'medicine', 'medical', 'doctor', 'disease', 'symptom',
        'treatment', 'well-being', 'ophthalmology', 'eye', 'vision', 'glaucoma',
        'diabetes', 'cataract', 'hypertension', 'myopia', 'AMD', 'amd', 'macular degeneration',
        'infection', 'inflammation', 'surgery', 'surgical', 'diagnosis', 'prescription',
        'therapy', 'clinic', 'hospital', 'pharmacy', 'medication', 'drug', 'illness',
        'condition', 'chronic', 'acute', 'recovery', 'rehabilitation', 'disorder',
        'consultation', 'appointment', 'specialist', 'physician', 'nurse', 'allergy',
        'immune system', 'immunity', 'pathology', 'anatomy', 'biopsy', 'imaging',
        'x-ray', 'MRI', 'CT scan', 'ultrasound', 'blood pressure', 'heart rate',
        'pulse', 'respiration', 'nutrition', 'diet', 'exercise', 'physical therapy',
        'mental health', 'psychiatry', 'psychology', 'counseling', 'wellness',
        'screening', 'preventive care', 'vaccination', 'immunization', 'epidemic',
        'pandemic', 'virus', 'bacteria', 'microorganism', 'pathogen', 'infection',
        'fever', 'pain', 'headache', 'migraine', 'cancer', 'tumor', 'oncology',
        'cardiology', 'neurology', 'dermatology', 'orthopedics', 'pediatrics',
        'gerontology', 'geriatrics', 'gastroenterology', 'urology', 'nephrology',
        'endocrinology', 'hematology', 'gynecology', 'obstetrics', 'radiology',
        'pulmonology', 'rheumatology', 'infectious disease', 'sore throat', 'cough',
        'flu', 'cold', 'COVID', 'virus', 'antibiotic', 'antiviral', 'antifungal',
        'blood test', 'cholesterol', 'glucose', 'insulin', 'metabolism', 'gene',
        'genetic', 'biological', 'biomedical', 'cell', 'tissue', 'organ', 'body',
        'muscle', 'bone', 'joint', 'arthritis', 'tendon', 'ligament', 'fracture',
        'sprain', 'strain', 'wound', 'laceration', 'bruise', 'burn', 'rash',
        'eczema', 'psoriasis', 'acne', 'dermatitis', 'skin', 'hair', 'nail',
        'hormone', 'thyroid', 'adrenal', 'pituitary', 'corticosteroid',
        'anti-inflammatory', 'analgesic', 'painkiller', 'sedative', 'anesthetic',
        'antiseptic', 'disinfectant', 'sterilization', 'hygiene', 'sanitation',
        'clinical trial', 'study', 'research', 'vaccine', 'booster', 'injection',
        'IV', 'intravenous', 'syringe', 'needle', 'scalpel', 'stethoscope',
        'blood pressure cuff', 'thermometer', 'sphygmomanometer', 'otoscope',
        'ophthalmoscope', 'reflex hammer', 'medical chart', 'record', 'report',
        'laboratory', 'specimen', 'sample', 'swab', 'culture', 'petri dish',
        'microscope', 'slide', 'pathogen', 'symptomatic', 'asymptomatic', 'outbreak',
        'quarantine', 'isolation', 'healthcare', 'health provider', 'clinical',
        'telemedicine', 'telehealth', 'consultation', 'triage', 'first aid', 'CPR',
        'emergency', 'urgent care', 'ambulance', 'paramedic', 'ER', 'ICU', 'sutures',
        'stitches', 'bandage', 'cast', 'splint', 'crutches', 'wheelchair', 'prosthesis',
        'orthotic', 'hearing aid', 'eyeglasses', 'contact lenses', 'vision correction',
        'LASIK', 'refractive surgery', 'eye exam', 'fundus', 'retina', 'cornea',
        'pupil', 'iris', 'lens', 'optic nerve', 'sclera', 'tear duct', 'vitreous humor', 'kids', 'adults',
        'elderly', 'infants', 'toddlers', 'ocular', 'diagnosis', 'prognosis', 'care', 'children', 'child', 'patient'
        # Add any more relevant keywords as necessary
    ]
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in health_keywords)

@app.route('/')
def index():
    # Example of setting a value in the session
    session['key'] = 'value'
    return 'Session set!'

# Define your image classification models
class DiseaseSpecificModel(nn.Module):
    def __init__(self, num_classes=8):
        super(DiseaseSpecificModel, self).__init__()
        # Use a pre-trained ResNet50 model
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Modify the output layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        return self.model(x)

# Load and configure the two-image classification model
two_image_model_path = 'newModel.pth'  # Path to the two-image model
two_image_model = DiseaseSpecificModel(num_classes=8)
two_image_model.load_state_dict(torch.load(two_image_model_path, map_location=torch.device('cpu')))
two_image_model.eval()

# Load and configure the single-image classification model
single_image_model_path = 'single_Model.pth'  # Path to the single-image model
single_image_model = DiseaseSpecificModel(num_classes=8)
single_image_model.load_state_dict(torch.load(single_image_model_path, map_location=torch.device('cpu')))
single_image_model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    'Normal': 'Your eyes appear to be normal. Maintain regular eye check-ups and a healthy lifestyle.',
    'Diabetes': 'Manage blood sugar levels, regular eye exams, possible laser treatment or surgery.',
    'Glaucoma': 'Medications, laser treatment, or surgery to lower intraocular pressure.',
    'Cataract': 'Surgery to replace the cloudy lens with an artificial lens.',
    'Age-related Macular Degeneration (AMD)': 'Anti-VEGF injections, laser therapy, lifestyle changes (diet, smoking cessation).',
    'Hypertension': 'Blood pressure control, lifestyle changes, medications.',
    'Myopia': 'Corrective lenses, orthokeratology, or refractive surgery for severe cases.',
    'Other Diseases/Abnormalities': 'Consult with an ophthalmologist for specific treatment options.'
}

# Map diseases to diagram image filenames
disease_diagram = {
    'Diabetes': 'diabetes.jpg',
    'Glaucoma': 'glaucoma.jpg',
    'Cataract': 'cataract.jpg',
    'Age-related Macular Degeneration (AMD)': 'amd.png',
    'Hypertension': 'hypertension.jpg',
    'Myopia': 'myopia.jpg',
    'Other Diseases/Abnormalities': 'Other_Diseases.jpg'
}

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Function to get the LLM response using Groq
def get_llm_response(disease_list, treatment_list, chat_history=None, is_image_upload=False):
    # Simplified system message
    system_message = (
        "You are a helpful and knowledgeable medical assistant specialized in ophthalmology."
        "Provide factual answers from a medical perspective."
        "Ensure the information is accurate and easy to understand."
        "You must not answer queries unrelated to health, medicine, or overall well-being."
    )

    # Prepare messages for chat completion
    messages = [{"role": "system", "content": system_message}]

    # Include chat history, limiting to the last 5 messages
    if chat_history:
        messages.extend(chat_history[-5:])

    # Generate the prompt based on the diagnosis
    if is_image_upload:
        if disease_list:
            if disease_list == ['Normal']:
                # Handle case where 'Normal' is predicted
                prompt = (
                    "The patient's eye images appear normal with no detectable signs of ocular diseases. "
                    "Provide guidelines for maintaining optimal eye health, including lifestyle recommendations."
                )
            else:
                diseases = ', '.join(disease_list)
                prompt = (
                    f"A patient is diagnosed with {diseases}. "
                    "Provide a detailed medical report that includes recommended treatment options, lifestyle changes, possible outcomes, and future prognosis."
                )
        else:
            # Handle case where neither 'Normal' nor diseases are predicted
            prompt = (
                "The patient shows no detectable signs of ocular diseases. "
                "Provide guidelines for maintaining optimal eye health, including lifestyle recommendations."
            )
        messages.append({"role": "user", "content": prompt})
    elif chat_history:
        # If not an image upload, rely on the existing chat history
        pass
    else:
        return "No input provided."

    print(f"Generated Messages: {messages}")  # For debugging

    # Generate text using the Groq API
    try:
        response = client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192"
        )
        response_text = response.choices[0].message.content
        print(f"LLM Response: {response_text}")  # For debugging
    except Exception as e:
        print(f"Error during text generation: {e}")
        return "Error during text generation."

    return response_text

# Custom function to generate a cache key based on the request body (prompt)
def make_cache_key():
    if request.json and 'prompt' in request.json:
        prompt = request.json['prompt'].strip().lower()  # Normalize prompt
        return md5(prompt.encode('utf-8')).hexdigest()
    return None

@app.route('/api/chat', methods=['POST'])
@cache.cached(timeout=300, key_prefix=make_cache_key)  # Cache using a custom key based on the prompt
def handle_chat():
    if request.json and 'prompt' in request.json:
        user_prompt = request.json['prompt']

        # Check if the prompt is health-related
        if not is_health_related(user_prompt):
            return jsonify({"response": "I can only assist with medical and health-related inquiries."}), 200

        session['chat_history'] = session.get('chat_history', [])
        session['chat_history'].append({'role': 'user', 'content': user_prompt})

        # Get the LLM response with chat history
        llm_response = get_llm_response([], [], chat_history=session['chat_history'])

        # Append the LLM response to the chat history with role 'assistant'
        session['chat_history'].append({'role': 'assistant', 'content': llm_response})

        return jsonify({"response": llm_response, "chat_history": session['chat_history']})
    
    return jsonify({"error": "Invalid input"}), 400

# API endpoint for handling file upload
@app.route('/api/upload', methods=['POST'])
def handle_upload():
    # Extract the user message if available
    message = request.form.get('message', '')

    # Check if the message is health-related if present
    if message and not is_health_related(message):
        return jsonify({"response": "I can only assist with medical and health-related inquiries."}), 200

    # Get the list of uploaded files
    uploaded_files = request.files.getlist('file')
    total_files = len(uploaded_files)

    # Count the number of images uploaded
    num_images = len(request.files)

    # Check if more than 2 images are uploaded
    if num_images > 2:
        return jsonify({"error": "Please upload no more than two images (left eye and right eye)."}), 400

    if num_images == 0:
        return jsonify({"error": "Please upload at least one eye image."}), 400

    # Handle file uploads
    left_file = request.files.get('left_eye', None)
    right_file = request.files.get('right_eye', None)

    # Ensure that only 'left_eye' and 'right_eye' are accepted
    accepted_filenames = ['left_eye', 'right_eye']
    for filename in request.files:
        if filename not in accepted_filenames:
            return jsonify({"error": f"Invalid file field '{filename}'. Only 'left_eye' and 'right_eye' are accepted."}), 400

    if left_file:
        left_filename = secure_filename(left_file.filename)
        left_filepath = os.path.join(app.config['UPLOAD_FOLDER'], left_filename)
        left_file.save(left_filepath)
    else:
        left_filepath = None

    if right_file:
        right_filename = secure_filename(right_file.filename)
        right_filepath = os.path.join(app.config['UPLOAD_FOLDER'], right_filename)
        right_file.save(right_filepath)
    else:
        right_filepath = None

    # Open and preprocess the images
    try:
        images = []
        if left_file:
            left_image = Image.open(left_filepath).convert('RGB')
            left_image = transform(left_image)
            images.append(left_image)
        if right_file:
            right_image = Image.open(right_filepath).convert('RGB')
            right_image = transform(right_image)
            images.append(right_image)
    except Exception as e:
        return jsonify({"error": f"Error processing images: {e}"}), 400

    # Determine which model to use based on the number of images
    if len(images) == 2:
        # Use the two-image model
        combined_image = torch.stack(images).mean(dim=0)
        combined_image = combined_image.unsqueeze(0)  # Add batch dimension

        # Predict diseases using the two-image model
        with torch.no_grad():
            outputs = two_image_model(combined_image)
            preds = torch.sigmoid(outputs).squeeze(0)  # Shape: (num_classes,)

    elif len(images) == 1:
        # Use the single-image model
        single_image = images[0].unsqueeze(0)  # Add batch dimension

        # Predict diseases using the single-image model
        with torch.no_grad():
            outputs = single_image_model(single_image)
            preds = torch.sigmoid(outputs).squeeze(0)  # Shape: (num_classes,)

    else:
        return jsonify({"error": "No valid images provided."}), 400

    # Apply threshold to predictions
    threshold = 0.5

    # Get indices of predicted disease classes (excluding 'Normal' class)
    disease_indices = list(range(1, len(class_to_disease)))  # indices 1-7
    predicted_disease_indices = (preds[disease_indices] > threshold).nonzero(as_tuple=True)[0].tolist()
    predicted_disease_indices = [i+1 for i in predicted_disease_indices]  # Adjust indices back to original

    # Check if 'Normal' is predicted
    normal_pred = preds[0] > threshold

    if predicted_disease_indices:
        predicted_diseases = [class_to_disease[idx] for idx in predicted_disease_indices]
        treatment_texts = [disease_treatment.get(disease, 'No treatment available') for disease in predicted_diseases]
    elif normal_pred:
        predicted_diseases = ['Normal']
        treatment_texts = [disease_treatment.get('Normal', 'Maintain regular eye check-ups and a healthy lifestyle.')]
    else:
        # Neither 'Normal' nor any diseases are predicted above threshold
        predicted_diseases = []
        treatment_texts = ['No specific treatment required. Maintain regular eye check-ups and a healthy lifestyle.']

    # Generate LLM response
    diagnosis_text = ', '.join(predicted_diseases) if predicted_diseases else 'No detectable diseases'
    treatment_text = '\n'.join(treatment_texts)

    # Get diagram URLs for diagnosed diseases
    diagram_urls = []
    for disease in predicted_diseases:
        diagram_filename = disease_diagram.get(disease)
        if diagram_filename:
            diagram_url = url_for('static', filename=f'diagrams/{diagram_filename}', _external=True)
            diagram_urls.append(diagram_url)

    # Update chat history
    session['chat_history'] = session.get('chat_history', [])
    user_message = f"Uploaded images diagnosed with: {diagnosis_text}"
    session['chat_history'].append({'role': 'user', 'content': user_message})

    # Get the LLM response with chat history
    llm_response = get_llm_response(predicted_diseases, treatment_texts, chat_history=session['chat_history'], is_image_upload=True)

    # Append the LLM response to the chat history
    session['chat_history'].append({'role': 'assistant', 'content': llm_response})

    # Include diagram URLs in the response
    return jsonify({
        "diagnosis": llm_response,
        "chat_history": session['chat_history'],
        "diagrams": diagram_urls
    })

if __name__ == '__main__':
    app.run(debug=True)
