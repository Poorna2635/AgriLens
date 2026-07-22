import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import re
import gc
import numpy as np
import tensorflow as tf

# Limit TensorFlow thread pool to prevent RAM spikes on cloud hosts
try:
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(2)
except Exception:
    pass

from flask import Flask, request, render_template, redirect, url_for, flash, session, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image

# Initialize the Flask app
app = Flask(__name__)

# Enable CORS for external frontend applications (e.g. Vercel deployment)
CORS(app)

# Configuration from Environment Variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'agrilens-production-secret-key-change-me')

# Ensure Upload Directory Exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

import threading

# TFLite Model Singleton (Lightweight: ~60 MB RAM usage)
tflite_path = os.path.join(BASE_DIR, 'Team3model.tflite')

interpreter = None
input_details = None
output_details = None
model_lock = threading.Lock()

def get_interpreter():
    global interpreter, input_details, output_details
    if interpreter is not None:
        return interpreter, input_details, output_details

    with model_lock:
        if interpreter is None:
            if os.path.exists(tflite_path):
                file_size = os.path.getsize(tflite_path)
                print(f"⚡ Loading TFLite model into RAM from {tflite_path} ({file_size/(1024*1024):.1f} MB)...")
                try:
                    interp = tf.lite.Interpreter(model_path=tflite_path)
                    interp.allocate_tensors()
                    inp_det = interp.get_input_details()
                    out_det = interp.get_output_details()

                    interpreter = interp
                    input_details = inp_det
                    output_details = out_det
                    print(f"✅ TFLite Interpreter ready in RAM!")
                except Exception as e:
                    print(f"❌ Error initializing TFLite Interpreter: {e}")
            else:
                print(f"⚠️ Warning: TFLite model file not found at {tflite_path}")

    return interpreter, input_details, output_details

def _warmup_interpreter():
    try:
        get_interpreter()
    except Exception as err:
        print(f"⚠️ Background interpreter warmup note: {err}")

threading.Thread(target=_warmup_interpreter, daemon=True).start()

img_width, img_height = 256, 256

# Class labels (38 classes)
class_labels = [
    'Bell Pepper-bacterial spot', 'Bell Pepper-healthy', 'Cassava-Bacterial Blight (CBB)',
    'Cassava-Brown Streak Disease (CBSD)', 'Cassava-Green Mottle (CGM)', 'Cassava-Healthy',
    'Cassava-Mosaic Disease (CMD)', 'Corn-cercospora leaf spot gray leaf spot', 'Corn-common rust',
    'Corn-healthy', 'Corn-northern leaf blight', 'Grape-black rot', 'Grape-esca (black measles)',
    'Grape-healthy', 'Grape-leaf blight (isariopsis leaf spot)', 'Mango-Anthracnose Fungal Leaf Disease',
    'Mango-Healthy Leaf', 'Mango-Rust Leaf Disease', 'Potato-early blight', 'Potato-healthy',
    'Potato-late blight', 'Rice-BrownSpot', 'Rice-Healthy', 'Rice-Hispa', 'Rice-LeafBlast',
    'Rose-Healthy Leaf', 'Rose-Rust', 'Rose-sawfly slug', 'Tomato-bacterial spot', 'Tomato-early blight',
    'Tomato-healthy', 'Tomato-late blight', 'Tomato-leaf mold', 'Tomato-mosaic virus',
    'Tomato-septoria leaf spot', 'Tomato-spider mites two-spotted spider mite', 'Tomato-target spot',
    'Tomato-yellow leaf curl virus'
]

# English Precaution Mapping
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

# Native Telugu Precaution Mapping
precaution_map_te = {
    'Bell Pepper-bacterial spot': "Fertilizer: ఎక్కువ నైట్రోజన్ నివారించండి. ఆకుల కణజాలాన్ని బలోపేతం చేయడానికి పొటాషియం సల్ఫేట్‌ను వాడండి.<br><br>Tip: మొక్క ఒత్తిడికి గురికాకుండా నిరోధించడానికి కాల్సియం నైట్రేట్ ద్రావణాన్ని పిచికారీ చేయండి.",
    'Bell Pepper-healthy': "సేంద్రీయ పదార్థాన్ని నిర్వహించడానికి కంపోస్ట్ లేదా పెరటి ఎరువును ఉపయోగించండి, ఇది వ్యాధిजनकాలతో సహజంగా పోరాడే నేల సూక్ష్మజీవులకు తోడ్పడుతుంది.",
    'Cassava-Bacterial Blight (CBB)': "Fertilizer: మ్యూరియేట్ ఆఫ్ పొటాష్ (MOP) ను ఉపయోగించండి. అధిక పొటాషియం వ్యాధి తీవ్రతను తగ్గిస్తుంది.<br><br>Tip: నేల బాగా నీరు పారేలా ఉండాలి. నీరు నిలిచే నేల వేళ్లను బలహీనపరుస్తుంది.",
    'Cassava-Brown Streak Disease (CBSD)': "Fertilizer: మ్యూరియేట్ ఆఫ్ పొటాష్ (MOP) ను ఉపయోగించండి.<br><br>Tip: నేల నీరు నిలవకుండా నిరోధించండి. తెల్లదోమ జనాభాను నియంత్రించండి.",
    'Cassava-Green Mottle (CGM)': "జీవ నియంత్రణలు లేదా పురుగుమందులను ఉపయోగించి పచ్చ పురుగులను నియంత్రించండి. నిరోధక రకాలను వాడండి.",
    'Cassava-Healthy': "సేంద్రీయ పదార్థాన్ని నిర్వహించడానికి కంపోస్ట్ లేదా పెరటి ఎరువును ఉపయోగించండి.",
    'Cassava-Mosaic Disease (CMD)': "Fertilizer: మ్యూరియేట్ ఆఫ్ పొటాష్ (MOP) ను ఉపయోగించండి.<br><br>Tip: నేల బాగా నీరు పారేలా ఉండాలి. తెల్లదోమలను నియంత్రించండి.",
    'Corn-cercospora leaf spot gray leaf spot': "Fertilizer: ప్రారంభ దశలలో మాత్రమే నైట్రోజన్ (యూరియా) వాడండి. జింక్ మరియు మెగ్నీషియం పిచికారీ చేయండి.<br><br>Tip: నేలలో ఫంగల్ జీవన చక్రం తెగడానికి పంట మార్పిడి చేయండి.",
    'Corn-common rust': "Fertilizer: ప్రారంభ దశలలో నైట్రోజన్ వాడండి. జింక్ మరియు మెగ్నీషియం పిచికారీ చేయండి.<br><br>Tip: పప్పుధాన్యాలతో పంట మార్పిడి చేయండి.",
    'Corn-healthy': "ప్రస్తుత ఎరువులు మరియు నీటి షెడ్యూల్‌ను కొనసాగించండి. క్రమం తప్పకుండా పొలాన్ని పర్యవేక్షించండి.",
    'Corn-northern leaf blight': "Fertilizer: ప్రారంభ దశలలో నైట్రోజన్ వాడండి. జింక్ మరియు మెగ్నీషియం పిచికారీ చేయండి.<br><br>Tip: పంట మార్పిడి చేయండి.",
    'Grape-black rot': "Fertilizer: బోరాన్ మరియు రాగి సూక్ష్మపోషకాలను వాడండి.<br><br>Tip: వర్షం తర్వాత ఆకులు త్వరగా ఆరిపోవడానికి లోపలి కొమ్మలను కత్తిరించండి.",
    'Grape-esca (black measles)': "Fertilizer: వేరు ఆరోగ్యానికి హ్యూమిక్ యాసిడ్ వాడండి.<br><br>Tip: తడి వాతావరణంలో కొమ్మల కత్తిరింపులు నివారించండి.",
    'Grape-healthy': "సరైన తీగ దూరాలను నిర్వహించండి మరియు సాధారణ పర్యవేక్షణ కొనసాగించండి.",
    'Grape-leaf blight (isariopsis leaf spot)': "Fertilizer: బోరాన్ మరియు రాగి సూక్ష్మపోషకాలను వాడండి.<br><br>Tip: లోపలి కొమ్మలను కత్తిరించండి.",
    'Mango-Anthracnose Fungal Leaf Disease': "Fertilizer: బోరాన్ మరియు కాపర్ సూక్ష్మపోషకాలను ఉపయోగించండి.<br><br>Tip: సూర్యరశ్మి మరియు గాలి తగిలేలా కొమ్మలను కత్తిరించండి.",
    'Mango-Healthy Leaf': "సాధారణ పర్యవేక్షణ మరియు సమతుల్య ఎరువులను కొనసాగించండి.",
    'Mango-Rust Leaf Disease': "Fertilizer: బోరాన్ మరియు కాపర్ సూక్ష్మపోషకాలను ఉపయోగించండి.<br><br>Tip: కొమ్మలను కత్తిరించండి.",
    'Potato-early blight': "Fertilizer: భాస్వరం సమృద్ధిగా ఉన్న ఎరువులను వాడండి.<br><br>Tip: పొడి వాతావరణంలో హార్వెస్టింగ్ చేయండి.",
    'Potato-healthy': "సాధారణ పర్యవేక్షణ మరియు సమతుల్య ఎరువులను కొనసాగించండి.",
    'Potato-late blight': "Fertilizer: భాస్వరం ఉన్న ఎరువులను వాడండి.<br><br>Tip: ఇది వేగంగా వ్యాపిస్తుంది - త్వరగా నివారణ చర్యలు తీసుకోండి.",
    'Rice-BrownSpot': "Fertilizer: పొటాషియం (K) మరియు మాంగనీస్ ఉపయోగించండి. పోషకాలు ఉన్న నేలలో ఈ వ్యాధి రాదు.",
    'Rice-Healthy': "సరైన నీటి మట్టాలను నిర్వహించండి మరియు సాధారణ పర్యవేక్షణ చేయండి.",
    'Rice-Hispa': "దెబ్బతిన్న ఆకుల చివరలను తొలగించండి. వేప ఆధారిత క్రిమిసంహారకాలను వాడండి.",
    'Rice-LeafBlast': "Fertilizer: నైట్రోజన్ వాడకం వెంటనే నిలిపివేయండి.<br><br>Tip: పొలంలో స్థిరమైన నీటి మట్టాన్ని నిర్వహించండి.",
    'Rose-Healthy Leaf': "సాధారణ పర్యవేక్షణ కొనసాగించండి.",
    'Rose-Rust': "Fertilizer: పొటాషియం ఎక్కువగా ఉన్న రోజ్ ఫుడ్ వాడండి.<br><br>Tip: వేప నూనెను పిచికారీ చేయండి.",
    'Rose-sawfly slug': "Fertilizer: పొటాషియం ఉన్న ఎరువులు వాడండి.<br><br>Tip: వేప నూనె ఉపయోగించండి.",
    'Tomato-bacterial spot': "Fertilizer: ఎక్కువ నైట్రోజన్ నివారించండి. పొటాషియం సల్ఫేట్‌ను వాడండి.<br><br>Tip: కాల్సియం నైట్రేట్ పిచికారీ చేయండి.",
    'Tomato-early blight': "Fertilizer: సమతుల్య NPK మరియు సిలికా ఉపయోగించండి.<br><br>Tip: మొక్క మొదలులో మల్చింగ్ చేయండి.",
    'Tomato-healthy': "ప్రస్తుత ఎరువులు మరియు నీటి షెడ్యూల్‌ను కొనసాగించండి.",
    'Tomato-late blight': "Fertilizer: NPK మరియు సిలికా వాడండి.<br><br>Tip: మొదలులో మల్చింగ్ చేయండి. త్వరగా స్పందించండి.",
    'Tomato-leaf mold': "Fertilizer: NPK మరియు సిలికా ఉపయోగించండి.<br><br>Tip: గాలి వెలుతురు పెంచండి.",
    'Tomato-mosaic virus': "Fertilizer: సముద్రపు పాచి సారాన్ని ఉపయోగించండి.<br><br>Tip: సోకిన మొక్కలను వెంటనే తొలగించండి.",
    'Tomato-septoria leaf spot': "Fertilizer: పొటాషియం సల్ఫేట్‌ను వాడండి.<br><br>Tip: కాల్సియం నైట్రేట్ పిచికారీ చేయండి.",
    'Tomato-spider mites two-spotted spider mite': "తేమను పెంచండి మరియు పురుగుమందులను ఉపయోగించండి.",
    'Tomato-target spot': "Fertilizer: పొటాషియం సల్ఫేట్ ఉపయోగించండి.<br><br>Tip: కాల్సియం నైట్రేట్ ద్రావణాన్ని వాడండి.",
    'Tomato-yellow leaf curl virus': "Fertilizer: సముద్రపు పాచి సారాన్ని ఉపయోగించండి.<br><br>Tip: సోకిన మొక్కలను తొలగించండి."
}

def get_precaution(label, lang='en'):
    if lang == 'te':
        return precaution_map_te.get(label, precaution_map.get(label, "సూచనలు లభ్యం కాలేదు."))
    return precaution_map.get(label, "No specific precaution found. Please consult with an agricultural expert for guidance.")

# Multi-language translation support (English & Telugu)
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
        'sign_in_google': 'Sign in with Google',
        'login_public_note': '💡 Public Access: You can enter any email & password to sign in instantly!',
        'enter_email': 'Enter any email (e.g. user@example.com)',
        'enter_password': 'Enter any password',
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
        'switch_camera': 'Switch Camera',
        'capture_photo': 'Capture Photo',
        'retake_photo': 'Retake Photo',
        'use_this_photo': 'Use This Photo',
        'camera_instructions': 'Place the leaf inside the frame for best results',
        'camera_not_supported': 'Camera is not supported on your device',
        'camera_permission_denied': 'Camera permission denied. Please allow camera access.'
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
        'sign_in_google': 'గూగుల్‌తో సైన్ ఇన్ చేయండి',
        'login_public_note': '💡 పబ్లిక్ ప్రవేశం: మీరు ఏ ఇమెయిల్ & పాస్‌వర్డ్‌నైనా ఉపయోగించి తక్షణమే లాగిన్ అవ్వవచ్చు!',
        'enter_email': 'ఏదైనా ఇమెయిల్ నమోదు చేయండి',
        'enter_password': 'ఏదైనా పాస్‌వర్డ్ నమోదు చేయండి',
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
        'fertilizer_recommendations': 'ఎరువుల సిఫార్సులు',
        'treatment_tips': 'చికిత్స చిట్కాలు',
        'fertilizer': 'ఎరువు:',
        'tip': 'చిట్కా:',
        'no_fertilizer': 'నిర్దిష్ట ఎరువుల సిఫార్సు అందుబాటులో లేదు.',
        'no_tips': 'నిర్దిష్ట చిట్కాలు అందుబాటులో లేవు. దయచేసి వ్యవసాయ నిపుణుడిని సంప్రదించండి.',
        'analyze_another': 'మరొక చిత్రాన్ని విశ్లేషించండి',
        'take_photo': 'నేరుగా ఫోటో తీయండి',
        'switch_camera': 'కెమెరా మార్చండి',
        'capture_photo': 'ఫోటో తీయండి',
        'retake_photo': 'మళ్లీ తీయండి',
        'use_this_photo': 'ఈ ఫోటోను ఉపయోగించండి',
        'camera_instructions': 'ఉత్తమ ఫలితాల కోసం ఆకును ఫ్రేమ్ లోపల ఉంచండి',
        'camera_not_supported': 'మీ పరికరంలో కెమెరా మద్దతు లేదు',
        'camera_permission_denied': 'కెమెరా అనుమతి నిరాకరించబడింది. దయచేసి కెమెరా ప్రాప్యతను అనుమతించండి.'
    }
}

def get_language():
    return session.get('lang', 'en')

def t(key):
    lang = get_language()
    return translations.get(lang, translations['en']).get(key, key)

def split_precaution(precaution_text):
    fertilizer = ""
    tips = ""
    parts = precaution_text.split('<br><br>')
    for part in parts:
        part_clean = part.strip()
        if part_clean.startswith('Fertilizer:') or part_clean.startswith('ఎరువు:'):
            fertilizer = re.sub(r'^(Fertilizer:|ఎరువు:)', '', part_clean).strip()
        elif part_clean.startswith('Tip:') or part_clean.startswith('చిట్కా:'):
            tips = re.sub(r'^(Tip:|చిట్కా:)', '', part_clean).strip()
        elif not fertilizer and part_clean:
            fertilizer = part_clean

    if not fertilizer and not tips:
        if 'Tip:' in precaution_text or 'చిట్కా:' in precaution_text:
            marker = 'Tip:' if 'Tip:' in precaution_text else 'చిట్కా:'
            idx = precaution_text.find(marker)
            fertilizer = re.sub(r'^(Fertilizer:|ఎరువు:)', '', precaution_text[:idx]).strip()
            tips = precaution_text[idx + len(marker):].strip()
        else:
            fertilizer = re.sub(r'^(Fertilizer:|ఎరువు:)', '', precaution_text).strip()

    fertilizer = re.sub(r'<[^>]+>', '', fertilizer).strip()
    tips = re.sub(r'<[^>]+>', '', tips).strip()
    return fertilizer, tips

def model_prediction(test_image_path):
    interp, inp_det, out_det = get_interpreter()
    if interp is None:
        raise ValueError("TFLite Model is not loaded. Ensure Team3model.tflite is present.")
    
    image = Image.open(test_image_path).convert('RGB')
    image = image.resize((img_width, img_height))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr], dtype=np.float32) / 255.0

    # Set input tensor and run lightweight TFLite inference
    interp.set_tensor(inp_det[0]['index'], input_arr)
    interp.invoke()
    predictions = interp.get_tensor(out_det[0]['index'])

    result = np.argmax(predictions[0])
    del input_arr, image
    gc.collect()
    return result

# Healthcheck Endpoint for Cloud Monitoring & Railway Health Checks
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'interpreter_ready': interpreter is not None
    }), 200

@app.route('/')
def login_redirect():
    session['logged_in'] = True
    return redirect(url_for('home'))

@app.route('/set-language/<lang>')
def set_language(lang):
    if lang in translations:
        session['lang'] = lang
    return redirect(request.referrer or url_for('home'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    session['logged_in'] = True
    if request.method == 'POST':
        return redirect(url_for('home'))
    return render_template('login.html', t=t, lang=get_language())

@app.route('/home')
def home():
    session['logged_in'] = True
    return render_template('home.html', t=t, lang=get_language())

@app.route('/disease-recognition', methods=['GET', 'POST'])
def disease_recognition():
    session['logged_in'] = True
    current_lang = get_language()
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part uploaded.')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected.')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
            except Exception as save_err:
                flash(f'File save failed: {str(save_err)}')
                return redirect(request.url)
            
            try:
                print(f"📸 Running TFLite prediction for: {filepath}")
                result_index = model_prediction(filepath)
                prediction = class_labels[result_index]
                precaution = get_precaution(prediction, lang=current_lang)
                fertilizer, tips = split_precaution(precaution)
                print(f"✅ Successful TFLite prediction: {prediction}")
                return render_template(
                    'prediction.html',
                    predicted_disease=prediction,
                    fertilizer=fertilizer,
                    tips=tips,
                    image_url=url_for('static', filename='uploads/' + filename),
                    t=t,
                    lang=current_lang
                )
            except Exception as e:
                print(f"❌ Prediction error: {e}")
                flash(f'Prediction failed: {str(e)}')
                return redirect(request.url)

    return render_template('disease-recognition.html', t=t, lang=current_lang)

# REST API Endpoint for Headless / Decoupled Frontend (e.g. Vercel)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            current_lang = request.args.get('lang', 'en')
            result_index = model_prediction(filepath)
            prediction = class_labels[result_index]
            precaution = get_precaution(prediction, lang=current_lang)
            fertilizer, tips = split_precaution(precaution)
            return jsonify({
                'success': True,
                'prediction': prediction,
                'fertilizer': fertilizer,
                'tips': tips,
                'image_url': f"/static/uploads/{filename}"
            }), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() in ['true', '1', 't']
    app.run(host=host, port=port, debug=debug)