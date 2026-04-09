

# 🛒 Real-Time Customer Time Tracking System

This project is a **computer vision-based system** that tracks customers in a store and measures how long each person stays inside.

It uses **YOLO (for detecting people)** and **BoT-SORT (for tracking them across frames)** to assign a unique ID to every customer and calculate their time in the store.

---

## 🚀 What This Project Does (In Simple Words)

* Detects people in a video (live camera or recorded video)
* Assigns each person a **unique ID**
* Tracks them as they move around
* Calculates **how long each person stays**
* Shows everything visually on the screen

👉 Example:

* Person 1 → 12 seconds
* Person 2 → 25 seconds

---

## 🧠 Technologies Used

* **YOLO v11m** → Detects people in each frame
* **BoT-SORT** → Tracks people and keeps IDs consistent
* **OpenCV** → Handles video processing and display
* **PyTorch** → Runs deep learning models
* **CUDA (optional)** → Makes it faster using GPU

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/customer-time-tracking.git
cd customer-time-tracking
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔧 Configuration

Before running the project:

* Create a `config.yaml` file (for camera or system settings)
* Adjust parameters in `settings.py` (like confidence threshold, tracking settings)
* Put your video inside the `streams/` folder

---

## ▶️ How to Run

### 🔴 For Live Camera

```bash
python time_notebook.py --source 0
```

---

### 🎥 For Video File

```bash
python time_notebook.py --source streams/your_video.mp4
```

---

## 🔍 How It Works (Step-by-Step)

### 1. Model Initialization

* Loads YOLO model
* Enables GPU (if available)
* Starts video capture
* Initializes BoT-SORT tracker

---

### 2. Frame Processing Loop

For every frame:

* Detects people using YOLO
* Sends detections to tracker
* Tracker assigns IDs
* Updates time for each ID
* Draws bounding boxes + ID + time

---

### ⏱️ Time Tracking Logic

```python
if track_id not in tracker_time:
    tracker_time[track_id] = [perf_counter(), perf_counter()]

tracker_time[track_id][1] = perf_counter()
```

👉 Meaning:

* First value = entry time
* Second value = latest seen time
* Time spent = difference between both

---

## 🎯 Features

✅ Real-time person detection
✅ Unique ID tracking
✅ Time calculation per person
✅ Live visualization with bounding boxes
✅ Works with webcam or video
✅ Optional video saving

---

## 📦 Requirements

* Python 3.x
* OpenCV
* Ultralytics YOLO
* PyTorch
* NumPy
* EasyDict

### Install manually if needed:

```bash
pip install opencv-python ultralytics torch numpy easydict
```

---

## 📊 Output

The system shows:

* Bounding boxes around people
* Unique ID (e.g., ID: 3)
* Time spent (e.g., 18.5 sec)

---

## ⚠️ Limitations

* Needs **good lighting**
* Can struggle in **crowded scenes**
* Tracking may break if people overlap
* Performance depends on GPU

---

## 🔮 Future Improvements

* Multi-camera tracking system
* Heatmaps of customer movement
* Zone-based analytics (which area people spend more time)
* Store data in database (for analysis)
* Customer behavior insights

---

## 📌 Use Cases

* Retail stores
* Shopping malls
* Customer behavior analysis
* Queue monitoring
* Store optimization

---

## 👨‍💻 About

This project demonstrates how **AI + Computer Vision** can be used in real-world retail environments to understand customer behavior without manual tracking.

---

## ⭐ Support

If you like this project:

* Star ⭐ the repository
* Fork 🍴 and improve it


