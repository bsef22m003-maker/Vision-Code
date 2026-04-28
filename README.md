# Vision-Code

![Build](https://img.shields.io/badge/build-passing-brightgreen)
![Backend](https://img.shields.io/badge/backend-Django%20%7C%20DRF-blue)
![CV Service](https://img.shields.io/badge/CV%20service-FastAPI-teal)
![Database](https://img.shields.io/badge/database-PostgreSQL%20%7C%20Supabase-blueviolet)
![AI](https://img.shields.io/badge/AI-ML%20%7C%20Computer%20Vision-orange)
![Frontend](https://img.shields.io/badge/frontend-Vercel-black)
![Backend Hosting](https://img.shields.io/badge/backend-Railway-purple)
![CI/CD](https://img.shields.io/badge/CI%2FCD-Git%20Triggered-success)
![Status](https://img.shields.io/badge/status-Completed-success)

---

## Project Description

**Vision-Code** is an AI-powered adaptive e-learning platform that personalizes learning experiences using **Computer Vision**, and **data-driven analytics**. The platform supports role-based access for students, instructors, and administrators, enabling secure authentication, structured course delivery, intelligent content recommendations, and automated proctoring during assessments.

The system follows a **microservices architecture** with independent deployment of the frontend, backend API, and Computer Vision service, and was developed using **Agile Scrum methodology**.

---

## Live Demo Link: https://www.vision-code.me/

---

## Features

### Core System

* Streamlined exam integrity and proctoring for graded exams
* marketplace for students to enroll courses
* course publication by instructors
* Secure user registration and login (Student / Instructor / Admin)
* Role-based authentication and authorization
* Session and token management
* Responsive landing and dashboard interfaces

### Learning Management

* Student panel with:
  * Article-based learning system
  * Courses, modules, and structured content
  * Quiz panel with automated evaluation
* Instructor panel with:
  * Course creation and publishing
  * Module and content management

### AI & Computer Vision

* Attention and proctoring module:
  * Eye gaze attention tracking
  * Face recognition
  * Multiple face detection
  * Sound detection during assessments

---

## System Architecture

### Microservices Overview

Vision-Code follows a **microservices architecture** with three independently deployed services communicating over REST:

| Service | Technology | Deployment |
|---|---|---|
| Frontend | HTML, CSS, JavaScript | Vercel |
| Backend API | Django + DRF | Railway |
| CV Service | FastAPI + OpenCV | Railway |
| Database | PostgreSQL | Supabase |

### Architecture Diagram

```
[ Web Frontend ]  ──────────────────────  Vercel
        |
        v (REST API)
[ Django REST API ]  ───────────────────  Railway
        |
        ├──> [ PostgreSQL (Supabase) ]
        |
        ├──> [ ML Recommendation Engine ]
        |
        └──> [ CV Proctoring Service ]  ─  Railway
                  (FastAPI + OpenCV)
```

### Service Descriptions

1. **Frontend Service**
   - Web-based interfaces for Students, Instructors, and Admins
   - Deployed on Vercel with automatic preview and production deployments

2. **Backend API Service**
   - Django + Django REST Framework
   - Handles authentication, business logic, course management, and data APIs
   - Deployed on Railway

3. **Computer Vision Service**
   - Standalone FastAPI microservice wrapping OpenCV-based CV models
   - Exposes REST endpoints consumed by the Backend API
   - Deployed independently on Railway for isolated scaling

4. **Database Layer**
   - Cloud-hosted PostgreSQL via Supabase
   - Stores user data, course content, quiz results, and logs

---

## CI/CD Pipeline

Deployments are fully automated and **git-triggered**:

- Pushing to `main` triggers production deployments on both Vercel (frontend) and Railway (backend + CV service)
- Pushing to feature branches triggers preview deployments on Vercel
- No manual deployment steps required after merge

---

## Technologies Used

### Backend
* Django
* Django REST Framework

### Computer Vision Service
* FastAPI
* OpenCV
* Python ML models (recommendation + proctoring)

### Frontend
* HTML, CSS, JavaScript

### Database
* PostgreSQL (Supabase)

### Infrastructure & DevOps
* Vercel (frontend hosting)
* Railway (backend + CV service hosting)
* Git-triggered CI/CD

---

## Local Development Setup

### Prerequisites

* Python 3.10+
* Git
* Node.js (if running frontend tooling locally)

### Clone the Repository

```bash
git clone https://github.com/org/vision-code.git
cd vision-code
```

### Backend API Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Create a `.env` file:

```env
DATABASE_URL=your_supabase_postgres_url
SECRET_KEY=your_django_secret_key
DEBUG=True
CV_SERVICE_URL=http://localhost:8001
```

Apply migrations and run:

```bash
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

Backend available at `http://127.0.0.1:8000/`

### Computer Vision Service Setup

```bash
cd cv-service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

CV service available at `http://127.0.0.1:8001/`

### Frontend Setup

Open `frontend/index.html` directly in a browser, or serve with any static file server.

---

## Deployment

| Service | Platform | Trigger |
|---|---|---|
| Frontend | Vercel | Push to `main` |
| Backend API | Railway | Push to `main` |
| CV Service | Railway | Push to `main` |

Environment variables for each service are configured via the respective platform dashboards (Vercel and Railway).

---

## Project Status

- **Sprint 1:** Core system, authentication ✔
- **Sprint 2:**   LMS foundation, AI/CV prototypes ✔
- **Sprint 3:**   integration of services ✔
- **Deployment:** Frontend on Vercel, Backend + CV Service on Railway ✔
- **CI/CD:** Git-triggered automated pipelines ✔
- **Status: Complete** ✔

---

## License

This project is developed for PUCIT for academic and research purposes.
