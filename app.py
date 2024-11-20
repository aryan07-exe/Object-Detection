from flask import Flask, request, render_template
from ultralytics import YOLO
import os

app = Flask(__name__)

model = YOLO('yolov8n.pt')  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file:
        # Save the uploaded image
        img_path = os.path.join('static/uploaded_images', file.filename)
        file.save(img_path)
        
        # Perform prediction using YOLO model
        results = model.predict(source=img_path, save=False, show=False)
        
        # Extract results as text
        detected_objects = []
        for result in results[0].boxes:
            xmin, ymin, xmax, ymax = result.xyxy[0].cpu().numpy()
            confidence = result.conf[0].cpu().numpy()
            class_id = int(result.cls[0].cpu().numpy())
            class_name = model.names[class_id]  # Get the class name from the model
            detected_objects.append(f"Class: {class_name}, Confidence: {confidence:.2f}, Bounding Box: ({xmin:.2f}, {ymin:.2f}, {xmax:.2f}, {ymax:.2f})")

        # Return the result as a text output
        return render_template('index.html', detected_objects=detected_objects)

if __name__ == '__main__':
    app.run(debug=True)