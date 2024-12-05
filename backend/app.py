from flask import Flask, request, jsonify, session, make_response, url_for
import torch
from PIL import Image
from torchvision import transforms
import os
import logging
import torch.nn as nn
import torchvision.models as models
from groq import Groq
from dotenv import load_dotenv
from flask_cors import CORS
from werkzeug.utils import secure_filename
from flask_caching import Cache
from hashlib import md5
from bs4 import BeautifulSoup
from flask import send_from_directory

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy import inspect
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import json


# Load environment variables from the .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "http://localhost:3000"}})
app.secret_key = os.getenv('FLASK_SECRET_KEY')  # Replace with a secure key
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'  # Adjust your database URI as needed
db = SQLAlchemy(app)
migrate = Migrate(app, db)

login_manager = LoginManager(app)
cache = Cache(config={'CACHE_TYPE': 'simple'})  # Simple in-memory cache
cache.init_app(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/diagrams/<filename>')
def diagram_file(filename):
    return send_from_directory('diagrams/', filename)



# Define the User, Chat, and Message models
class User(db.Model, UserMixin):
    __tablename__ = 'user'  # Explicitly define the table name
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    password_hash = db.Column(db.String(128))
    chats = db.relationship('Chat', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
  
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False, default='New Chat')
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    messages = db.relationship('Message', backref='chat', lazy=True, cascade='all, delete-orphan')


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    role = db.Column(db.String(10), nullable=False)
    content = db.Column(db.Text, nullable=False)
    chat_id = db.Column(db.Integer, db.ForeignKey('chat.id'), nullable=False)
    # The backref 'chat' in Chat model already provides access to the Chat object
    images = db.Column(db.Text)  # Store image URLs as JSON string
    diagrams = db.Column(db.Text)  # Store diagram URLs as JSON string

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Error handler for 404 Not Found
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Resource not found.'}), 404

# Error handler for 403 Forbidden
@app.errorhandler(403)
def forbidden_error(error):
    return jsonify({'error': 'Unauthorized access.'}), 403

# Error handler for 500 Internal Server Error
@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'An internal server error occurred.'}), 500

# Function to check if a prompt is health-related
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
        'elderly', 'infants', 'toddlers', 'ocular', 'diagnosis', 'prognosis', 'care', 'children', 'child', 'patient',
        'causes', 'genetic', 'hereditary', 'natural', 'carcinogens', 'food', 'sugar', 'age'
        # Add any more relevant keywords as necessary
    ]
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in health_keywords)

# Route for user registration
@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided.'}), 400

        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return jsonify({'error': 'Username and password are required.'}), 400

        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already exists.'}), 400

        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        login_user(new_user)

        return jsonify({'message': 'Registration successful.'}), 201

    except Exception as e:
        logging.exception("Exception during registration")
        return jsonify({'error': 'Internal server error.'}), 500

# Route for user login
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        login_user(user)
        return jsonify({'message': 'Login successful.'}), 200
    else:
        return jsonify({'error': 'Invalid credentials.'}), 401
    
@login_manager.unauthorized_handler
def unauthorized_callback():
    # Return a JSON response with a 401 Unauthorized status
    return jsonify({'error': 'Unauthorized'}), 401

# Route for user logout
@app.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'message': 'Logged out successfully.'}), 200

# Route to create a new chat
@app.route('/api/chats', methods=['POST'])
@login_required
def create_chat():
    try:
        # Parse incoming JSON
        data = request.get_json()
        if not data or 'name' not in data:
            return jsonify({'error': 'Chat name is required.'}), 400

        chat_name = data.get('name', 'New Chat')

        # Create a new chat
        new_chat = Chat(name=chat_name, user_id=current_user.id)
        db.session.add(new_chat)
        db.session.commit()

        # Respond with the new chat details
        return jsonify({'chat_id': new_chat.id, 'name': new_chat.name}), 201
    except Exception as e:
        # Handle unexpected errors
        return jsonify({'error': 'Failed to create chat.', 'message': str(e)}), 500


# Route to get all chats for the current user
@app.route('/api/chats', methods=['GET'])
@login_required
def get_chats():
    try:
        chats = Chat.query.filter_by(user_id=current_user.id).all()
        chat_list = [{'id': chat.id, 'name': chat.name} for chat in chats]
        return jsonify(chat_list), 200
    except Exception as e:
        app.logger.error(f"Error in /api/chats: {str(e)}")
        return jsonify({'error': 'Failed to fetch chats'}), 500

@app.route('/api/chats/<int:chat_id>/messages', methods=['GET'])
@login_required
def get_chat_messages(chat_id):
    chat = Chat.query.get_or_404(chat_id)

    if chat.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized access.'}), 403

    messages = Message.query.filter_by(chat_id=chat_id).all()
    messages_data = []

    for message in messages:
        message_data = {
            'id': message.id,
            'role': message.role,
            'content': message.content,
            'images': json.loads(message.images) if message.images else [],
            'diagrams': json.loads(message.diagrams) if message.diagrams else [],
        }

        if message.images:
            message_data['images'] = json.loads(message.images)
        if message.diagrams:
            message_data['diagrams'] = json.loads(message.diagrams)

        messages_data.append(message_data)

    return jsonify(messages_data), 200

# Route to delete a chat
@app.route('/api/chats/<int:chat_id>', methods=['DELETE'])
@login_required
def delete_chat(chat_id):
    chat = Chat.query.get_or_404(chat_id)

    if chat.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized access.'}), 403

    db.session.delete(chat)
    db.session.commit()

    return jsonify({'message': 'Chat deleted successfully.'}), 200

# Load and configure models, transformations, and mappings as before
# (FundusDetector, DiseaseSpecificModel, class_to_disease, disease_treatment, disease_diagram)

# Define the model
class FundusDetector(nn.Module):
    def __init__(self):
        super(FundusDetector, self).__init__()
        # Load a pretrained ResNet50 model
        self.model = models.resnet50(pretrained=True)
        # Replace the last fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 2)
        )

    def forward(self, x):
        return self.model(x)

# Load and configure the fundus detector model
fundus_detector_model_path = 'best_fundus_detector.pth'  # Path to the fundus detector model
fundus_detector = FundusDetector()
fundus_detector.load_state_dict(torch.load(fundus_detector_model_path, map_location=torch.device('cpu')))
fundus_detector.eval()


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

# Load and configure the dual-image classification model (processes single images)
dual_image_model_path = 'newDualModel.pth'  # Path to the new dual-image model
dual_image_model = DiseaseSpecificModel(num_classes=8)
dual_image_model.load_state_dict(torch.load(dual_image_model_path, map_location=torch.device('cpu')))
dual_image_model.eval()

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

# Updated function to get the LLM response using Groq
# Updated function to get the LLM response using Groq
def get_llm_response(left_diseases=None, right_diseases=None, chat_history=None, is_image_upload=False):
    # Simplified system message
    system_message = (
        "You are a helpful and knowledgeable medical assistant specialized in ophthalmology. "
        "Provide factual answers from a medical perspective. "
        "Ensure the information is accurate and easy to understand. "
        # "Present the information in plain text without any HTML or markup. "
        "You must not answer queries unrelated to health, medicine, or overall well-being."
    )

    # Prepare messages for chat completion
    messages = [{"role": "system", "content": system_message}]

    # Include chat history, limiting to the last 5 messages
    if chat_history:
        messages.extend(chat_history[-5:])

    # Generate the prompt based on the diagnosis
    if is_image_upload:
        # Determine which eyes have diagnoses
        prompt_parts = []
        eye_descriptions = []

        if left_diseases is not None:
            left_diseases_str = ', '.join(left_diseases) if left_diseases else 'No detectable diseases'
            prompt_parts.append(f"- Left Eye: {left_diseases_str}")
            if left_diseases == ['Normal']:
                eye_descriptions.append("left eye")

        if right_diseases is not None:
            right_diseases_str = ', '.join(right_diseases) if right_diseases else 'No detectable diseases'
            prompt_parts.append(f"- Right Eye: {right_diseases_str}")
            if right_diseases == ['Normal']:
                eye_descriptions.append("right eye")

        # Check if all provided eyes are normal
        if (
            (left_diseases == ['Normal'] if left_diseases is not None else True) and
            (right_diseases == ['Normal'] if right_diseases is not None else True)
        ):
            if eye_descriptions:
                eye_str = " and ".join(eye_descriptions)
                prompt = (
                    f"The patient's {'eyes are' if len(eye_descriptions) > 1 else 'eye is'} normal with no detectable signs of ocular diseases "
                    f"in the {eye_str}. "
                    "Provide guidelines for maintaining optimal eye health, including lifestyle recommendations."
                )
            else:
                # No eyes provided or no diagnoses, default message
                prompt = (
                    "No detectable diseases were found in the uploaded eye image. "
                    "Provide guidelines for maintaining optimal eye health, including lifestyle recommendations."
                )
        else:
            prompt = (
                "A patient is diagnosed with the following condition(s):\n"
                f"{chr(10).join(prompt_parts)}\n"
                "Provide a detailed medical report that includes recommended treatment options, lifestyle changes, possible outcomes, and future prognosis for each condition."
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

        # Remove HTML tags from the response using BeautifulSoup
        soup = BeautifulSoup(response_text, 'html.parser')
        clean_response_text = soup.get_text()
        print(f"Cleaned LLM Response: {clean_response_text}")  # For debugging

    except Exception as e:
        print(f"Error during text generation: {e}")
        return "Error during text generation."

    return clean_response_text


# Route to add a message and get the LLM response
@app.route('/api/chats/<int:chat_id>/messages', methods=['POST'])
@login_required
def add_message(chat_id):
    chat = Chat.query.get_or_404(chat_id)
    if chat.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized.'}), 403

    data = request.get_json()
    user_prompt = data.get('prompt')

    if user_prompt:
        # Check if the prompt is health-related
        if not is_health_related(user_prompt):
            return jsonify({"response": "I can only assist with medical and health-related inquiries."}), 200

        # Append the user message to the chat history in the database
        new_message = Message(role='user', content=user_prompt, chat_id=chat.id)
        db.session.add(new_message)
        db.session.commit()

        # Retrieve the last 5 messages for context
        messages = Message.query.filter_by(chat_id=chat.id).order_by(Message.id).all()
        chat_history = [{'role': msg.role, 'content': msg.content} for msg in messages][-5:]

        # Get the LLM response
        llm_response = get_llm_response(chat_history=chat_history)

        # Append the assistant's response to the chat history in the database
        assistant_message = Message(role='assistant', content=llm_response, chat_id=chat.id)
        db.session.add(assistant_message)
        db.session.commit()

        return jsonify({"response": llm_response, "chat_history": chat_history})

    return jsonify({"error": "Invalid input"}), 400

# Route to get messages from a chat
@app.route('/api/chats/<int:chat_id>/messages', methods=['GET'])
@login_required
def get_messages(chat_id):
    chat = Chat.query.get_or_404(chat_id)
    if chat.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized.'}), 403

    messages = Message.query.filter_by(chat_id=chat.id).all()
    message_list = [{'role': msg.role, 'content': msg.content} for msg in messages]
    return jsonify(message_list), 200

# API endpoint for handling file upload
@app.route('/api/chats/<int:chat_id>/upload', methods=['POST'])
@login_required
def handle_upload(chat_id):
    chat = Chat.query.get_or_404(chat_id)
    if chat.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized.'}), 403

    # Extract the user message if available
    message_text = request.form.get('message', '')

    # Check if the message is health-related if present
    if message_text and not is_health_related(message_text):
        return jsonify({"response": "I can only assist with medical and health-related inquiries."}), 200

    # Get the list of uploaded files
    uploaded_files = request.files.getlist('file')
    eye_labels = request.form.getlist('eye_labels')

    # Validate number of images and eye labels
    if len(uploaded_files) != len(eye_labels):
        return jsonify({"error": "The number of images does not match the number of eye labels."}), 400

    # Validate eye labels
    valid_eye_labels = ['left', 'right']
    for label in eye_labels:
        if label.lower() not in valid_eye_labels:
            return jsonify({"error": f"Invalid eye label '{label}'. Accepted values are 'left' or 'right'."}), 400

    if len(uploaded_files) == 0:
        return jsonify({"error": "Please upload at least one eye image."}), 400

    if len(uploaded_files) > 2:
        return jsonify({"error": "Please upload no more than two images (left eye and right eye)."}), 400

    # Process uploaded files and eye labels
    images = {}
    image_urls = []
    for file, eye_label in zip(uploaded_files, eye_labels):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Generate image URL and add to list
        image_url = url_for('uploaded_file', filename=filename, _external=True)
        image_urls.append({'url': image_url, 'eye_label': eye_label.lower()})

        # Open and preprocess the image
        try:
            image = Image.open(filepath).convert('RGB')
            image = transform(image)
            images[eye_label.lower()] = image
        except Exception as e:
            return jsonify({"error": f"Error processing image '{filename}': {e}"}), 400


    # Check if images are fundus images
    non_fundus_images = []
    for eye_label, image in images.items():
        image_tensor = image.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = fundus_detector(image_tensor)
            _, preds = torch.max(outputs, 1)
        if preds.item() == 1:  # Assuming class 1 is 'fundus', class 0 is 'non-fundus'
            non_fundus_images.append(eye_label)

    if non_fundus_images:
        # If any images are not fundus images, return an error message
        non_fundus_eyes = ' and '.join(non_fundus_images)
        return jsonify({"error": f"The uploaded image(s) for {non_fundus_eyes} eye(s) are not fundus images. EyeGPT can only analyze fundus images."}), 400

    # Initialize dictionaries to hold predictions
    predicted_diseases = {}
    treatment_texts = {}
    diagram_urls = {}

    # Threshold for predictions
    threshold = 0.5
    disease_indices = list(range(1, len(class_to_disease)))  # indices 1-7

    # Predict diseases for each eye
    for eye_label in images:
        image_tensor = images[eye_label].unsqueeze(0)  # Add batch dimension

        # Predict diseases using the single-image model
        with torch.no_grad():
            outputs = single_image_model(image_tensor)
            preds = torch.sigmoid(outputs).squeeze(0)  # Shape: (num_classes,)

        # Apply threshold to predictions
        pred_disease_indices = (preds[disease_indices] > threshold).nonzero(as_tuple=True)[0].tolist()
        pred_disease_indices = [i+1 for i in pred_disease_indices]  # Adjust indices back to original

        normal_pred = preds[0] > threshold

        if pred_disease_indices:
            diseases = [class_to_disease[idx] for idx in pred_disease_indices]
            treatments = [disease_treatment.get(disease, 'No treatment available') for disease in diseases]
        elif normal_pred:
            diseases = ['Normal']
            treatments = [disease_treatment.get('Normal', 'Maintain regular eye check-ups and a healthy lifestyle.')]
        else:
            diseases = []
            treatments = ['No specific treatment required. Maintain regular eye check-ups and a healthy lifestyle.']

        predicted_diseases[eye_label] = diseases
        treatment_texts[eye_label] = treatments

        # Get diagram URLs for diagnosed diseases
        diagrams = []
        for disease in diseases:
            diagram_filename = disease_diagram.get(disease)
            if diagram_filename:
                diagram_url = url_for('static', filename=f'diagrams/{diagram_filename}', _external=True)
                diagrams.append(diagram_url)
        diagram_urls[eye_label] = diagrams


    # After getting the diagnosis, update the chat history

    # Prepare the diagnosis text
    diagnosis_text = ""
    for eye_label in ['left', 'right']:
        if eye_label in predicted_diseases:
            diseases = predicted_diseases[eye_label]
            diagnosis = ', '.join(diseases) if diseases else 'No detectable diseases'
            diagnosis_text += f"{eye_label.capitalize()} Eye: {diagnosis}\n"

    # Save the user's message with images
    user_message = f"Uploaded images diagnosed with:\n{diagnosis_text}"
    new_message = Message(
        role='user',
        content=message_text if message_text else user_message,
        chat_id=chat.id,
        images=json.dumps(image_urls)  # Serialize image URLs
    )
    db.session.add(new_message)

    # Prepare left and right diseases, using None if the eye was not uploaded
    left_diseases = predicted_diseases.get('left', None)
    right_diseases = predicted_diseases.get('right', None)

    # Retrieve the last 5 messages for context
    messages = Message.query.filter_by(chat_id=chat.id).order_by(Message.id).all()
    chat_history = [{'role': msg.role, 'content': msg.content} for msg in messages][-5:]

    # Get the LLM response
    llm_response = get_llm_response(
        left_diseases=left_diseases,
        right_diseases=right_diseases,
        chat_history=chat_history,
        is_image_upload=True
    )

    # Collect all diagram URLs into a single list
    all_diagram_urls = []
    for eye_label in ['left', 'right']:
        all_diagram_urls.extend(diagram_urls.get(eye_label, []))

    # Save the assistant's response with the combined diagrams
    assistant_message = Message(
        role='assistant',
        content=llm_response,
        chat_id=chat.id,
        diagrams=json.dumps(all_diagram_urls)  # Serialize list of diagram URLs
    )
    db.session.add(assistant_message)
    db.session.commit()

    # Include image URLs and diagram URLs in the response
    response_data = {
        "diagnosis": llm_response,
        "image_urls": image_urls,
        "diagram_urls": [],
    }

    # Collect diagram URLs from left and right eyes
    for eye_label in ['left', 'right']:
        if eye_label in predicted_diseases:
            response_data[f'{eye_label}_eye'] = {
                "diagnosis": ', '.join(predicted_diseases[eye_label]) if predicted_diseases[eye_label] else 'No detectable diseases',
                "diagrams": diagram_urls.get(eye_label, [])
            }
            response_data['diagram_urls'].extend(diagram_urls.get(eye_label, []))

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
