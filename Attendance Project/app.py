from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import threading
import os
import cv2
import time
import numpy as np
import pickle
import csv
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# تحديد مسارات المجلدات والملفات
BASE_DIR = "D:/Attendance Project"
DATA_DIR = os.path.join(BASE_DIR, "DATA")
MODEL_FILE = os.path.join(BASE_DIR, "face_recognition_model.h5")
LABEL_ENCODER_FILE = os.path.join(BASE_DIR, "label_encoder.pkl")
LOG_FILE = os.path.join(BASE_DIR, "attendance_log.csv")
STUDENT_COUNT_FILE = os.path.join(BASE_DIR, "student_count.txt")
CSV_FILE = os.path.join(BASE_DIR, "user_data.csv")

# إنشاء التطبيق باستخدام Flask
app = Flask(__name__, static_folder='static')

# 1. دالة إنشاء مستخدم جديد
def create_user(username, user_id, subject, email, phone, gender, dob):
    """
    دالة لإنشاء مستخدم جديد في النظام مع جمع صور الوجه من كاميرا الويب.
    
    :param username: اسم المستخدم
    :param user_id: الرقم التعريفي للمستخدم
    :param subject: الموضوع الدراسي
    :param email: البريد الإلكتروني للمستخدم
    :param phone: رقم الهاتف
    :param gender: الجنس
    :param dob: تاريخ الميلاد
    :return: رسالة تؤكد النجاح أو الخطأ
    """
    folder_path = os.path.join(DATA_DIR, username)

    # التحقق من عدم وجود مستخدم بنفس الرقم التعريفي
    if os.path.isfile(CSV_FILE):
        with open(CSV_FILE, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row and row[1] == user_id:
                    return "Error: User with this ID already exists."

    # إنشاء المجلد لحفظ صور المستخدم
    os.makedirs(folder_path, exist_ok=True)

    # التحقق من وجود ملف عدد الطلاب
    if not os.path.isfile(STUDENT_COUNT_FILE):
        with open(STUDENT_COUNT_FILE, mode='w') as count_file:
            count_file.write('0')

    # فتح كاميرا الويب لالتقاط الصور
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    start_time = time.time()
    img_count = 0

    # التقاط صور الوجه
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            img_path = os.path.join(folder_path, f"{username}_{img_count}.jpg")
            cv2.imwrite(img_path, face)
            img_count += 1

        cv2.putText(frame, f"Images Captured: {img_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Capturing Faces', frame)

        # التوقف بعد 15 ثانية أو التقاط 150 صورة
        if time.time() - start_time > 15 or img_count >= 150:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # إضافة بيانات المستخدم إلى ملف CSV
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Username', 'UserID', 'Subject', 'Email', 'Phone', 'Gender', 'Date of Birth'])
        writer.writerow([username, user_id, subject, email, phone, gender, dob])

    # تحديث عدد الطلاب المسجلين
    with open(STUDENT_COUNT_FILE, mode='r') as count_file:
        student_count = int(count_file.read().strip())

    with open(STUDENT_COUNT_FILE, mode='w') as count_file_obj:
        count_file_obj.write(str(student_count + 1))

    return "User created successfully!"

# 2. دالة تدريب النموذج
def train_model():
    """
    دالة لتدريب نموذج التعرف على الوجوه باستخدام الصور الموجودة في المجلد DATA.
    """
    print("Starting training...")
    folders = os.listdir(DATA_DIR)
    images = []
    labels = []

    # التحقق من وجود بيانات للتدريب
    if not folders:
        raise ValueError("The DATA folder is empty. Please add class folders with images.")

    # تحميل الصور والبيانات
    for folder_name in folders:
        folder_path = os.path.join(DATA_DIR, folder_name)
        if os.path.isdir(folder_path):
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
                    img_equalized = cv2.equalizeHist(img_blurred)
                    img_resized = cv2.resize(img_equalized, (64, 64))
                    images.append(img_resized)
                    labels.append(folder_name)

    # معالجة البيانات لتدريب النموذج
    images = np.array(images).reshape(-1, 64, 64, 1) / 255.0
    labels = np.array(labels)

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)

    X_train, X_test, y_train, y_test = train_test_split(images, labels_categorical, test_size=0.2, random_state=42)

    # بناء النموذج
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])

    # تجميع النموذج
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # تدريب النموذج
    model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

    # تقييم النموذج
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # حفظ النموذج والـ Label Encoder
    model.save(MODEL_FILE)
    print("Model saved successfully.")
    with open(LABEL_ENCODER_FILE, 'wb') as file:
        pickle.dump(label_encoder, file)
    print("Label encoder saved successfully.")

# 3. دالة اختبار النموذج
webcam = None  # متغير عام للـ webcam

def test_model():
    """
    دالة لاختبار النموذج المدرب باستخدام كاميرا الويب وتحديد الأشخاص المتعرف عليهم.
    """
    global webcam
    print("Starting testing...")
    model = load_model(MODEL_FILE)
    print("Model loaded successfully.")

    with open(LABEL_ENCODER_FILE, 'rb') as file:
        label_encoder = pickle.load(file)
    print("Label encoder loaded successfully.")

    webcam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # إنشاء سجل الحضور
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Name', 'Time'])

    # تحميل السجلات السابقة
    existing_records = set()
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                existing_records.add(row[0])

    recognition_count = {}  # عدد مرات التعرف على كل شخص
    while webcam.isOpened():
        ret, frame = webcam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = gray[y:y+h, x:x+w]

            face_blurred = cv2.GaussianBlur(face, (5, 5), 0)
            face_equalized = cv2.equalizeHist(face_blurred)
            face_resized = cv2.resize(face_equalized, (64, 64)) / 255.0
            face_input = np.reshape(face_resized, (1, 64, 64, 1))

            predictions = model.predict(face_input)
            confidence = np.max(predictions)  # الثقة في التنبؤ

            if confidence < 0.9:  # إذا كانت الثقة أقل من 90%، تجاهل الشخص
                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("Unknown person detected with low confidence.")
                continue

            predicted_label_index = np.argmax(predictions)
            predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]

            # تتبع عدد مرات التعرف على كل شخص
            if predicted_label in recognition_count:
                recognition_count[predicted_label] += 1
            else:
                recognition_count[predicted_label] = 1

            # تسجيل الشخص فقط إذا تم التعرف عليه 100 مرة والثقة كافية
            if recognition_count[predicted_label] >= 100 and predicted_label not in existing_records:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with open(LOG_FILE, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([predicted_label, current_time])
                existing_records.add(predicted_label)
                print(f"Recorded: {predicted_label} after 100 detections with confidence {confidence:.2f}")

            # عرض اسم الشخص على الشاشة مع عدد مرات التعرف
            cv2.putText(frame, f"{predicted_label} ({recognition_count[predicted_label]})", 
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()

# مسارات Flask

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """
    عرض صفحة التسجيل للمستخدمين الجدد وإضافة مستخدم عند إرسال النموذج.
    """
    if request.method == 'POST':
        username = request.form['username']
        user_id = request.form['user_id']
        subject = request.form['subject']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        dob = request.form['dob']

        if not all([username, user_id, subject, email, phone, gender, dob]):
            return render_template('error.html', message="All fields are required.")

        result = create_user(username, user_id, subject, email, phone, gender, dob)
        if "Error:" in result:
            return render_template('error.html', message=result)
        return render_template('success.html', message=result)

    return render_template('register.html')

@app.route('/model')
def model_dashboard():
    try:
        return render_template('model.html')
    except Exception as e:
        return render_template('error.html', message="Failed to load model dashboard. Please try again.")

@app.route('/train', methods=['GET'])
def train():
    """
    بدء عملية التدريب للنموذج في خيط منفصل.
    """
    def train_task():
        try:
            train_model()
        except Exception as e:
            print(f"Error in training: {e}")
    
    # بدء التدريب في خيط منفصل
    train_thread = threading.Thread(target=train_task)
    train_thread.start()

    return render_template('train.html')

@app.route('/test', methods=['GET'])
def test():
    """
    بدء عملية اختبار النموذج للتعرف على الوجوه.
    """
    def test_task():
        try:
            test_model()
        except Exception as e:
            print(f"Error in testing: {e}")
    
    # بدء الاختبار في خيط منفصل
    test_thread = threading.Thread(target=test_task)
    test_thread.start()

    return render_template('test.html')

@app.route('/view', methods=['GET'])
def view_attendance():
    """
    عرض سجلات الحضور المخزنة في ملف CSV.
    """
    try:
        with open(LOG_FILE, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # تخطي رأس الجدول
            records = list(reader)

        return render_template('attendance.html', records=records)
    except Exception as e:
        return render_template('error.html', message="Error loading attendance records.")

@app.route('/stop_test', methods=['GET'])
def stop_test():
    """
    إيقاف اختبار النموذج.
    """
    global webcam
    if webcam:
        webcam.release()
        cv2.destroyAllWindows()
        webcam = None

    return render_template('index.html', message="Test stopped.")

# تشغيل التطبيق
if __name__ == "__main__":
    app.run(debug=True)
