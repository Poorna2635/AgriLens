import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import re
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from PIL import Image

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['SECRET_KEY'] = 'supersecretkey'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
model_path = 'Team3model.h5'
model = tf.keras.models.load_model(model_path, compile=False)

# Manually configure the model's loss function
model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(reduction='sum_over_batch_size'))

img_width, img_height = 256, 256

# Class labels
class_labels = ['Bell Pepper-bacterial spot', 'Bell Pepper-healthy', 'Cassava-Bacterial Blight (CBB)',
                'Cassava-Brown Streak Disease (CBSD)', 'Cassava-Green Mottle (CGM)', 'Cassava-Healthy',
                'Cassava-Mosaic Disease (CMD)', 'Corn-cercospora leaf spot gray leaf spot', 'Corn-common rust',
                'Corn-healthy', 'Corn-northern leaf blight', 'Grape-black rot', 'Grape-esca (black measles)',
                'Grape-healthy', 'Grape-leaf blight (isariopsis leaf spot)', 'Mango-Anthracnose Fungal Leaf Disease',
                'Mango-Healthy Leaf', 'Mango-Rust Leaf Disease', 'Potato-early blight', 'Potato-healthy',
                'Potato-late blight', 'Rice-BrownSpot', 'Rice-Healthy', 'Rice-Hispa', 'Rice-LeafBlast',
                'Rose-Healthy Leaf', 'Rose-Rust', 'Rose-sawfly slug', 'Tomato-bacterial spot', 'Tomato-early blight',
                'Tomato-healthy', 'Tomato-late blight', 'Tomato-leaf mold', 'Tomato-mosaic virus',
                'Tomato-septoria leaf spot', 'Tomato-spider mites two-spotted spider mite', 'Tomato-target spot',
                'Tomato-yellow leaf curl virus']

# Precaution mapping for each disease
precaution_map = {
    'Bell Pepper-bacterial spot': "Fertilizer: Avoid high Nitrogen. Apply Potassium Sulfate to strengthen leaf tissues.<br><br>Tip: Spray a Calcium Nitrate solution to prevent the plant from becoming stressed, which makes it more susceptible to spots.",
    'Bell Pepper-healthy': "Use Compost or Manure to maintain organic matter, which supports the beneficial soil microbes that fight pathogens naturally.",
    'Cassava-Bacterial Blight (CBB)': "Fertilizer: Apply Muriate of Potash (MOP). High potassium levels are proven to reduce the severity of Cassava diseases.<br><br>Tip: Ensure the soil is well-drained; waterlogged soil weakens Cassava roots and invites bacterial rot.",
    'Cassava-Brown Streak Disease (CBSD)': "Fertilizer: Apply Muriate of Potash (MOP). High potassium levels help reduce disease severity.<br><br>Tip: Ensure the soil is well-drained; waterlogged soil weakens Cassava roots. Control whitefly populations and use certified virus-free cuttings.",
    'Cassava-Green Mottle (CGM)': "Control green mite infestations using biological controls or miticides. Use resistant cultivars. Ensure well-drained soil.",
    'Cassava-Healthy': "Use Compost or Manure to maintain organic matter, which supports the beneficial soil microbes that fight pathogens naturally.",
    'Cassava-Mosaic Disease (CMD)': "Fertilizer: Apply Muriate of Potash (MOP). High potassium levels are proven to reduce the severity of Cassava Mosaic Disease.<br><br>Tip: Ensure the soil is well-drained; waterlogged soil weakens Cassava roots. Control whitefly populations and use virus-free cuttings.",
    'Corn-cercospora leaf spot gray leaf spot': "Fertilizer: Use a Top-dressing of Nitrogen (Urea) only during the early growth stages. For diseased plants, apply Zinc and Magnesium foliar sprays.<br><br>Tip: Rotate corn with legumes (beans/peas) next season to break the fungal life cycle in the soil.",
    'Corn-common rust': "Fertilizer: Use a Top-dressing of Nitrogen (Urea) only during the early growth stages. For diseased plants, apply Zinc and Magnesium foliar sprays.<br><br>Tip: Rotate corn with legumes (beans/peas) next season to break the fungal life cycle in the soil.",
    'Corn-healthy': "Maintain current fertilization and watering schedule. Continue regular field monitoring.",
    'Corn-northern leaf blight': "Fertilizer: Use a Top-dressing of Nitrogen (Urea) only during the early growth stages. For diseased plants, apply Zinc and Magnesium foliar sprays.<br><br>Tip: Rotate corn with legumes (beans/peas) next season to break the fungal life cycle in the soil.",
    'Grape-black rot': "Fertilizer: Apply Boron and Copper micronutrients. Copper isn't just a fertilizer; it acts as a fungicide.<br><br>Tip: Prune the inner branches of the vine to allow sunlight and wind to dry the leaves quickly after rain.",
    'Grape-esca (black measles)': "Fertilizer: Focus on soil conditioners like Humic Acid to improve root health.<br><br>Tip: Avoid heavy pruning during wet weather as the spores enter through open wounds.",
    'Grape-healthy': "Maintain proper vine spacing and continue routine monitoring. Ensure adequate air circulation.",
    'Grape-leaf blight (isariopsis leaf spot)': "Fertilizer: Apply Boron and Copper micronutrients.<br><br>Tip: Prune the inner branches of the vine to allow sunlight and wind to dry the leaves quickly after rain.",
    'Mango-Anthracnose Fungal Leaf Disease': "Fertilizer: Apply Boron and Copper micronutrients. Copper isn't just a fertilizer; it acts as a fungicide.<br><br>Tip: Prune the inner branches of the tree to allow sunlight and wind to dry the leaves quickly after rain.",
    'Mango-Healthy Leaf': "Continue routine scouting and balanced fertilization. Maintain good orchard hygiene.",
    'Mango-Rust Leaf Disease': "Fertilizer: Apply Boron and Copper micronutrients. Copper isn't just a fertilizer; it acts as a fungicide.<br><br>Tip: Prune the inner branches of the tree to allow sunlight and wind to dry the leaves quickly after rain.",
    'Potato-early blight': "Fertilizer: Use Phosphorus-rich fertilizers (like Bone Meal) to encourage strong tuber development even if the leaves are struggling.<br><br>Tip: Harvest during dry weather and ensure the tubers are 'cured' (skin hardened) before storage to prevent rot.",
    'Potato-healthy': "Continue routine scouting and balanced fertilization. Monitor for early disease signs.",
    'Potato-late blight': "Fertilizer: Use Phosphorus-rich fertilizers (like Bone Meal) to encourage strong tuber development even if the leaves are struggling.<br><br>Tip: Harvest during dry weather and ensure the tubers are 'cured' (skin hardened) before storage to prevent rot. This is highly destructive - act quickly.",
    'Rice-BrownSpot': "Fertilizer: This disease is an 'indicator' of poor soil. Apply Potassium (K) and Manganese. Brown spot rarely occurs in well-nourished soil.",
    'Rice-Healthy': "Maintain proper water levels and continue balanced fertilization. Monitor for early disease signs.",
    'Rice-Hispa': "Manual removal of the damaged leaf tips where larvae are present. Use neem-based pesticides or recommended insecticides.",
    'Rice-LeafBlast': "Fertilizer: STOP Nitrogen application immediately if you see Blast. Excess nitrogen makes the rice plant 'succulent' and very easy for the fungus to eat.<br><br>Tip: Maintain a consistent water level in the paddy to reduce plant stress.",
    'Rose-Healthy Leaf': "Continue routine scouting and balanced fertilization. Maintain good garden hygiene.",
    'Rose-Rust': "Fertilizer: Use a slow-release Rose food high in Potassium.<br><br>Tip: For Sawflies, use Neem Oil—it acts as both a pesticide and a leaf shine that prevents rust spores from sticking.",
    'Rose-sawfly slug': "Fertilizer: Use a slow-release Rose food high in Potassium.<br><br>Tip: Use Neem Oil—it acts as both a pesticide and a leaf shine that prevents rust spores from sticking.",
    'Tomato-bacterial spot': "Fertilizer: Avoid high Nitrogen. Apply Potassium Sulfate to strengthen leaf tissues.<br><br>Tip: Spray a Calcium Nitrate solution to prevent the plant from becoming stressed, which makes it more susceptible to spots.",
    'Tomato-early blight': "Fertilizer: Use a balanced NPK 10-10-10 but supplement with Silica. Silica creates a physical barrier on the leaf that fungi cannot easily penetrate.<br><br>Tip: Mulch around the base to prevent soil-borne spores from splashing onto the leaves during watering.",
    'Tomato-healthy': "Maintain current fertilization and watering schedule. Continue regular monitoring.",
    'Tomato-late blight': "Fertilizer: Use a balanced NPK 10-10-10 but supplement with Silica. Silica creates a physical barrier on the leaf that fungi cannot easily penetrate.<br><br>Tip: Mulch around the base to prevent soil-borne spores from splashing onto the leaves during watering. This spreads rapidly - act quickly.",
    'Tomato-leaf mold': "Fertilizer: Use a balanced NPK 10-10-10 but supplement with Silica. Silica creates a physical barrier on the leaf that fungi cannot easily penetrate.<br><br>Tip: Mulch around the base. Increase ventilation and keep humidity below 85%.",
    'Tomato-mosaic virus': "Fertilizer: There is no cure for the virus, but Seaweed Extract can help the plant tolerate the stress.<br><br>Tip: Immediately remove infected plants to save the rest of the crop. Control whiteflies (the carriers).",
    'Tomato-septoria leaf spot': "Fertilizer: Avoid high Nitrogen. Apply Potassium Sulfate to strengthen leaf tissues.<br><br>Tip: Spray a Calcium Nitrate solution to prevent the plant from becoming stressed, which makes it more susceptible to spots.",
    'Tomato-spider mites two-spotted spider mite': "Increase humidity (they prefer dry heat) and use miticides or predatory mites. Ensure adequate watering.",
    'Tomato-target spot': "Fertilizer: Avoid high Nitrogen. Apply Potassium Sulfate to strengthen leaf tissues.<br><br>Tip: Spray a Calcium Nitrate solution to prevent the plant from becoming stressed, which makes it more susceptible to spots.",
    'Tomato-yellow leaf curl virus': "Fertilizer: There is no cure for the virus, but Seaweed Extract can help the plant tolerate the stress.<br><br>Tip: Immediately remove infected plants to save the rest of the crop. Control whiteflies (the carriers)."
}

# Function to get precaution for a given disease label
def get_precaution(label):
    """
    Returns the recommended treatment or precaution for a given disease label.
    
    Args:
        label (str): The disease class label
        
    Returns:
        str: The precaution/treatment recommendation
    """
    return precaution_map.get(label, "No specific precaution found. Please consult with an agricultural expert for guidance.")

# Translation dictionaries
translations = {
    'en': {
        'app_name': 'AgriLens',
        'home': 'Home',
        'disease_recognition': 'Disease Recognition',
        'logout': 'Logout',
        'login_title': 'AgriLens',
        'email': 'Email',
        'password': 'Password',
        'login': 'Login',
        'enter_email': 'Enter your email',
        'enter_password': 'Enter your password',
        'welcome_title': 'Welcome to Intelligent Plant Disease Detection System Using Deep Learning 🌿🔍',
        'welcome_message': 'Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let\'s protect our crops and ensure a healthier harvest!',
        'how_it_works': 'How It Works',
        'upload_image': 'Upload Image:',
        'upload_image_desc': 'Go to the Disease Recognition page and upload an image of a plant with suspected diseases.',
        'analysis': 'Analysis:',
        'analysis_desc': 'Our system will process the image using advanced algorithms to identify potential diseases.',
        'results': 'Results:',
        'results_desc': 'View the results and recommendations for further action.',
        'features': 'Our Features',
        'feature1': 'High Accuracy: Our model is trained on a large dataset to ensure high accuracy in disease detection.',
        'feature2': 'Easy to Use: Upload an image and get results in seconds.',
        'feature3': 'Secure: Your data is safe with us. We prioritize your privacy.',
        'supported_plants': 'Supported Plants',
        'start_detection': 'Start Detection',
        'disease_recognition_title': 'Plant Disease Recognition',
        'disease_recognition_desc': 'Upload an image of a plant leaf to detect diseases and get treatment recommendations.',
        'select_image': 'Select Plant Image',
        'analyze': 'Analyze Plant Disease',
        'instructions': 'Instructions:',
        'instruction1': 'Use clear, well-lit photos of plant leaves',
        'instruction2': 'Ensure the leaf fills most of the image frame',
        'instruction3': 'Supported formats: JPG, PNG, JPEG',
        'health_report': 'Plant Health Report',
        'fertilizer_recommendations': 'Fertilizer Recommendations',
        'treatment_tips': 'Treatment Tips',
        'fertilizer': 'Fertilizer:',
        'tip': 'Tip:',
        'no_fertilizer': 'No specific fertilizer recommendation available.',
        'no_tips': 'No specific tips available. Please consult with an agricultural expert.',
        'analyze_another': 'Analyze Another Image',
        'take_photo': 'Take Photo Directly',
        'upload_image': 'Upload Image',
        'switch_camera': 'Switch Camera',
        'capture_photo': 'Capture Photo',
        'retake_photo': 'Retake Photo',
        'use_this_photo': 'Use This Photo',
        'camera_instructions': 'Place the leaf inside the frame for best results',
        'camera_not_supported': 'Camera is not supported on your device',
        'camera_permission_denied': 'Camera permission denied. Please allow camera access.',
    },
    'te': {
        'app_name': 'అగ్రిలెన్స్',
        'home': 'హోమ్',
        'disease_recognition': 'వ్యాధి గుర్తింపు',
        'logout': 'లాగ్అవుట్',
        'login_title': 'మొక్కల వ్యాధి వర్గీకరణ',
        'email': 'ఇమెయిల్',
        'password': 'పాస్వర్డ్',
        'login': 'లాగిన్',
        'enter_email': 'మీ ఇమెయిల్ నమోదు చేయండి',
        'enter_password': 'మీ పాస్వర్డ్ నమోదు చేయండి',
        'welcome_title': 'మొక్కల వ్యాధి గుర్తింపు వ్యవస్థకు స్వాగతం 🌿🔍',
        'welcome_message': 'మొక్కల వ్యాధులను సమర్థవంతంగా గుర్తించడంలో సహాయపడటం మా లక్ష్యం. మొక్క యొక్క చిత్రాన్ని అప్లోడ్ చేయండి, మా వ్యవస్థ దానిని విశ్లేషించి వ్యాధుల సంకేతాలను గుర్తిస్తుంది. కలిసి, మన పంటలను రక్షించి, ఆరోగ్యకరమైన పంటను నిర్ధారిద్దాం!',
        'how_it_works': 'ఇది ఎలా పని చేస్తుంది',
        'upload_image': 'చిత్రం అప్లోడ్ చేయండి:',
        'upload_image_desc': 'వ్యాధి గుర్తింపు పేజీకి వెళ్లి, అనుమానాస్పద వ్యాధులతో మొక్క యొక్క చిత్రాన్ని అప్లోడ్ చేయండి.',
        'analysis': 'విశ్లేషణ:',
        'analysis_desc': 'మా వ్యవస్థ అధునాతన అల్గోరిథమ్లను ఉపయోగించి చిత్రాన్ని ప్రాసెస్ చేసి, సంభావ్య వ్యాధులను గుర్తిస్తుంది.',
        'results': 'ఫలితాలు:',
        'results_desc': 'ఫలితాలు మరియు తదుపరి చర్యలకు సిఫార్సులను వీక్షించండి.',
        'features': 'మా లక్షణాలు',
        'feature1': 'అధిక ఖచ్చితత్వం: వ్యాధి గుర్తింపులో అధిక ఖచ్చితత్వాన్ని నిర్ధారించడానికి మా మోడల్ పెద్ద డేటాసెట్లో శిక్షణ పొందింది.',
        'feature2': 'ఉపయోగించడం సులభం: చిత్రాన్ని అప్లోడ్ చేసి సెకన్లలో ఫలితాలను పొందండి.',
        'feature3': 'సురక్షితం: మీ డేటా మాతో సురక్షితం. మేము మీ గోప్యతకు ప్రాధాన్యత ఇస్తాము.',
        'supported_plants': 'సమర్థించబడిన మొక్కలు',
        'start_detection': 'గుర్తింపు ప్రారంభించండి',
        'disease_recognition_title': 'మొక్కల వ్యాధి గుర్తింపు',
        'disease_recognition_desc': 'వ్యాధులను గుర్తించడానికి మరియు చికిత్స సిఫార్సులను పొందడానికి మొక్క ఆకు యొక్క చిత్రాన్ని అప్లోడ్ చేయండి.',
        'select_image': 'మొక్క చిత్రాన్ని ఎంచుకోండి',
        'analyze': 'మొక్క వ్యాధిని విశ్లేషించండి',
        'instructions': 'సూచనలు:',
        'instruction1': 'మొక్క ఆకుల యొక్క స్పష్టమైన, బాగా వెలుగుతున్న ఫోటోలను ఉపయోగించండి',
        'instruction2': 'ఆకు చిత్ర ఫ్రేమ్ యొక్క చాలా భాగాన్ని నింపుతుందని నిర్ధారించండి',
        'instruction3': 'సమర్థించబడిన ఫార్మాట్లు: JPG, PNG, JPEG',
        'health_report': 'మొక్క ఆరోగ్య నివేదిక',
        'fertilizer_recommendations': 'ఎరువు సిఫార్సులు',
        'treatment_tips': 'చికిత్స చిట్కాలు',
        'fertilizer': 'ఎరువు:',
        'tip': 'చిట్కా:',
        'no_fertilizer': 'నిర్దిష్ట ఎరువు సిఫార్సు అందుబాటులో లేదు.',
        'no_tips': 'నిర్దిష్ట చిట్కాలు అందుబాటులో లేవు. దయచేసి వ్యవసాయ నిపుణుడిని సంప్రదించండి.',
        'analyze_another': 'మరొక చిత్రాన్ని విశ్లేషించండి',
        'take_photo': 'నేరుగా ఫోటో తీయండి',
        'upload_image': 'చిత్రం అప్లోడ్ చేయండి',
        'switch_camera': 'కెమెరా మార్చండి',
        'capture_photo': 'ఫోటో తీయండి',
        'retake_photo': 'మళ్లీ తీయండి',
        'use_this_photo': 'ఈ ఫోటోను ఉపయోగించండి',
        'camera_instructions': 'ఉత్తమ ఫలితాల కోసం ఆకును ఫ్రేమ్ లోపల ఉంచండి',
        'camera_not_supported': 'మీ పరికరంలో కెమెరా మద్దతు లేదు',
        'camera_permission_denied': 'కెమెరా అనుమతి నిరాకరించబడింది. దయచేసి కెమెరా ప్రాప్యతను అనుమతించండి.',
    }
}

# # Function to get current language (always English)
# def get_language():
#     """Returns the current language, always 'en'"""
#     return 'en'
def get_language():
    return session.get('lang', 'en')

# Function to get translation
def t(key):
    """Returns translation for the given key in current language"""
    lang = get_language()
    return translations.get(lang, translations['en']).get(key, key)

# Function to split precaution into fertilizer and tips
def split_precaution(precaution_text):
    """
    Splits the precaution text into fertilizer and tips sections.
    
    Args:
        precaution_text (str): The full precaution text
        
    Returns:
        tuple: (fertilizer, tips) - Two strings containing fertilizer and tips
    """
    fertilizer = ""
    tips = ""
    
    # Split by <br><br> which separates sections
    parts = precaution_text.split('<br><br>')
    
    for part in parts:
        part_clean = part.strip()
        if part_clean.startswith('Fertilizer:'):
            fertilizer = part_clean.replace('Fertilizer:', '').strip()
        elif part_clean.startswith('Tip:'):
            tips = part_clean.replace('Tip:', '').strip()
        elif not fertilizer and part_clean:
            # If no fertilizer found yet and this part has content, treat as fertilizer
            fertilizer = part_clean
    
    # Fallback: if still not split, try direct string search
    if not fertilizer and not tips:
        if 'Tip:' in precaution_text:
            tip_index = precaution_text.find('Tip:')
            fertilizer = precaution_text[:tip_index].replace('Fertilizer:', '').strip()
            tips = precaution_text[tip_index + 4:].strip()
        else:
            fertilizer = precaution_text.replace('Fertilizer:', '').strip()
    
    # Clean up HTML tags from text
    fertilizer = re.sub(r'<[^>]+>', '', fertilizer).strip()
    tips = re.sub(r'<[^>]+>', '', tips).strip()
    
    return fertilizer, tips

# Function to predict the class of the plant disease
def model_prediction(test_image_path):
    image = Image.open(test_image_path)
    image = image.resize((img_width, img_height))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    input_arr = input_arr / 255.0
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

@app.route('/')
def login_redirect():
    return redirect(url_for('login'))

@app.route('/set-language/<lang>')
def set_language(lang):
    if lang in translations:
        session['lang'] = lang
    return redirect(request.referrer or url_for('home'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Bypass username and password validation
        session['logged_in'] = True
        return redirect(url_for('home'))
    return render_template('login.html', t=t, lang=get_language())

@app.route('/home')
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('home.html', t=t, lang=get_language())

@app.route('/disease-recognition', methods=['GET', 'POST'])
def disease_recognition():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
            except UnicodeEncodeError:
                flash('File name contains unsupported characters.')
                return redirect(request.url)
            result_index = model_prediction(filepath)
            prediction = class_labels[result_index]
            precaution = get_precaution(prediction)
            fertilizer, tips = split_precaution(precaution)
            return render_template('prediction.html', predicted_disease=prediction, fertilizer=fertilizer, tips=tips, image_url=url_for('static', filename='uploads/' + filename), t=t, lang=get_language())
    return render_template('disease-recognition.html', t=t, lang=get_language())

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)