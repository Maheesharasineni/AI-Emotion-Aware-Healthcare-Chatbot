# A real-time AI-powered healthcare assistant that detects user emotions through the webcam and adapts responses using a mental-health–aware LLM agent. Built using Flask, CNN Emotion Model, OpenCV, MongoDB Atlas, and Ngrok.

## What This Project Does
- Detects your facial emotion in real time  
- Shows the emotion and confidence  
- Gives supportive replies based on what you feel  
- Allows login & signup (MongoDB Atlas)  
- Runs fully on Google Colab using Ngrok  
- This project does not use HTML pages—it directly opens the chatbot page when the Flask app starts.

## Main Files in the Project
- **app.py** → Runs the chatbot  
- **camera.py** → Handles emotion detection  
- **emotion_model1.h5** → Trained model (stored in Google Drive)  
- **requirements.txt** → Required packages  
- **train.py** → Training script (optional)

## Important Links

### Google Colab Notebook (Run the full project online)  
**Colab Link:**  
https://colab.research.google.com/drive/1MZF-ELKFgMFQUkUBrzAuyH2BPMx2aUJs?usp=sharing  

### Trained Emotion Model (Google Drive Folder)  
**Model Link:**  
https://drive.google.com/drive/folders/1NczqJUuBqUm73rIPY2pkow7dfS14hlWm?usp=sharing  

### GitHub Repository  
https://github.com/Maheesharasineni/AI-Emotion-Aware-Healthcare-Chatbot  

## How to Run On Your Computer

### Install required packages:

pip install -r requirements.txt

### Download the model file from Google Drive  
Place **emotion_model1.h5** inside your project folder.

### Run the app:
## On Google Colab
1. Open the Colab link  
2. Mount Google Drive  
3. Load the model  
4. Start Flask  
5. Create public link using Ngrok  


## Created By
**Maheesha Rasineni**
