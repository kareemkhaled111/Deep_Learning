from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
import time
import csv

# إنشاء تطبيق Flask
app = Flask(__name__, static_folder='static')

# إعداد المسارات للملفات
BASE_DIR = "D:/Attendance Project/DATA"  # مسار مجلد البيانات (صور المستخدمين)
CSV_FILE = "D:/Attendance Project/user_data.csv"  # مسار ملف البيانات (الذي يحتوي على معلومات المستخدمين)
COUNT_FILE = "D:/Attendance Project/student_count.txt"  # مسار ملف عدد الطلاب (لتخزين عدد الطلاب المسجلين)

# وظيفة لإنشاء مستخدم جديد
def create_user(username, user_id, subject, email, phone, gender, dob):
    folder_path = os.path.join(BASE_DIR, username)  # مسار المجلد الخاص بالمستخدم

    # التحقق من وجود المستخدم مسبقًا بناءً على ID فقط
    if os.path.isfile(CSV_FILE):
        with open(CSV_FILE, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row and row[1] == user_id:
                    return "Error: User with this ID already exists."  # إذا كان المستخدم موجودًا، إرجاع رسالة خطأ

    # إنشاء مجلد للمستخدم إذا لم يكن موجودًا
    os.makedirs(folder_path, exist_ok=True)

    # إذا كان ملف count غير موجود، قم بإنشائه وتحديده كـ 0
    if not os.path.isfile(COUNT_FILE):
        with open(COUNT_FILE, mode='w') as count_file:
            count_file.write('0')

    # تشغيل الكاميرا لالتقاط صور المستخدم
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # تحميل كاشف الوجه

    start_time = time.time()  # بدء التوقيت لوقف التقاط الصور بعد وقت معين
    img_count = 0  # عداد الصور الملتقطة

    # حلقة لالتقاط الصور من الكاميرا
    while True:
        ret, frame = cap.read()  # قراءة الإطار من الكاميرا
        if not ret:
            break

        # تحويل الصورة إلى تدرج الرمادي لاكتشاف الوجوه
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        # رسم مستطيل حول الوجه المكتشف
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]  # استخراج الوجه من الصورة
            img_path = os.path.join(folder_path, f"{username}_{img_count}.jpg")  # تحديد مسار حفظ الصورة
            cv2.imwrite(img_path, face)  # حفظ الصورة
            img_count += 1

        # عرض عداد الصور الملتقطة في نافذة الكاميرا
        cv2.putText(frame, f"Images Captured: {img_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Capturing Faces', frame)

        # التوقف عن التقاط الصور بعد 15 ثانية أو 150 صورة
        if time.time() - start_time > 15 or img_count >= 150:
            break

        # إغلاق النافذة عند الضغط على 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # تحرير الكاميرا وإغلاق جميع النوافذ
    cap.release()
    cv2.destroyAllWindows()

    # تحديث ملف CSV لتخزين بيانات المستخدم
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Username', 'UserID', 'Subject', 'Email', 'Phone', 'Gender', 'Date of Birth'])  # إضافة عنوان الأعمدة إذا كان الملف فارغًا
        writer.writerow([username, user_id, subject, email, phone, gender, dob])  # إضافة بيانات المستخدم الجديدة

    # تحديث عدد الطلاب في ملف COUNT_FILE
    with open(COUNT_FILE, mode='r') as count_file:
        student_count = int(count_file.read().strip())

    with open(COUNT_FILE, mode='w') as count_file_obj:
        count_file_obj.write(str(student_count + 1))  # زيادة العدد بمقدار 1

    return "User created successfully!"  # إرجاع رسالة النجاح

# إعدادات واجهة الويب
@app.route('/')
def home():
    return render_template('index.html')  # الصفحة الرئيسية

# صفحة التسجيل للمستخدمين الجدد
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':  # إذا كان الطلب من نوع POST (إرسال البيانات)
        username = request.form['username']
        user_id = request.form['user_id']
        subject = request.form['subject']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        dob = request.form['dob']

        # التحقق من أن جميع الحقول قد تم ملؤها
        if not username or not user_id or not subject or not email:
            return render_template('error.html', message="All fields are required.")  # إظهار رسالة خطأ إذا كانت الحقول فارغة

        # إنشاء المستخدم
        result = create_user(username, user_id, subject, email, phone, gender, dob)
        return render_template('success.html', message=result)  # إظهار رسالة نجاح عند إتمام التسجيل

    return render_template('register.html')  # عرض صفحة التسجيل عند استخدام GET

# صفحة الأيقونة المفضلة
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static/images'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

# تشغيل التطبيق
if __name__ == '__main__':
    app.run(debug=True)  # تشغيل الخادم في وضع التصحيح
