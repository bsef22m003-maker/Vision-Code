# Vision-Code

![Build](https://img.shields.io/badge/build-active-brightgreen)
![Backend](https://img.shields.io/badge/backend-Django%20%7C%20DRF-blue)
![Database](https://img.shields.io/badge/database-PostgreSQL%20%7C%20Supabase-blueviolet)
![AI](https://img.shields.io/badge/AI-ML%20%7C%20Computer%20Vision-orange)
![Status](https://img.shields.io/badge/status-Sprint%201%20Completed-success)

---

## Project Description

**Vision-Code** is an AI-powered adaptive e-learning platform designed to personalize learning experiences using **Machine Learning**, **Computer Vision**, and **data-driven analytics**. The platform supports role-based access for students, instructors, and administrators, enabling secure authentication, structured course delivery, intelligent content recommendations, and automated proctoring during assessments.

The system is developed following **Agile Scrum methodology**, with incremental delivery across sprints.

---

## Features Implemented (Sprint 1)

### Core System

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

### AI & Computer Vision (Prototype)

* Attention and proctoring module:

  * Eye gaze attention tracking
  * Face recognition
  * Multiple face detection
  * Sound detection during assessments
* Machine Learning–based content recommendation system *(training & testing phase)*
* Adaptive learning roadmap recommendation system *(training & testing phase)*

---

## System Architecture

### High-Level Architecture Overview

The platform follows a **modular client–server architecture** with integrated AI services:

1. **Frontend Layer**

   * Web-based user interfaces for Students, Instructors, and Admins
   * Communicates with backend services via REST APIs

2. **Backend Layer**

   * Django + Django REST Framework
   * Handles authentication, business logic, course management, and APIs

3. **Database Layer**

   * Cloud-hosted PostgreSQL (Supabase)
   * Stores user data, course content, quiz results, and logs

4. **AI & ML Layer**

   * Recommendation engines for content and learning paths
   * Computer Vision services for attention tracking and proctoring

5. **External Services**

   * Supabase for managed PostgreSQL and scalability

---

## Architecture Diagram (Logical View)

```
[ Web Frontend ]
        |
        v
[ Django REST API ]
        |
        +--> [ PostgreSQL (Supabase) ]
        |
        +--> [ ML Recommendation Engine ]
        |
        +--> [ Computer Vision Proctoring Module ]
```

---

## Technologies Used

### Backend

* Django
* Django REST Framework

### Frontend

* HTML, CSS, JavaScript

### Database

* PostgreSQL (Supabase)

### AI / ML & Computer Vision

* Python
* OpenCV
* Machine Learning models for recommendation systems

---

## Setup Instructions

### Prerequisites

* Python 3.10+
* Git

### Clone the Repository

```bash
git clone https://github.com/org/vision-code.git
```

### Navigate to Project Directory

```bash
cd vision-code
```

### Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
DATABASE_URL=your_supabase_postgres_url
SECRET_KEY=your_django_secret_key
DEBUG=True
```

### Apply Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

### Run the Application

```bash
python manage.py runserver
```

Access the application at:

```
http://127.0.0.1:8000/
```

## Known Issues / TODOs

* Integration of computer vision proctoring into live assessments pending
* Recommendation systems under training and evaluation
* Instructor analytics dashboard not implemented
* Gamification (badges, streaks) pending
* Mobile responsiveness not fully tested
* OAuth / social login not implemented

---

## Project Status

* **Sprint 1:** Core system, authentication, LMS foundation, AI/CV prototypes ✔
* **Sprint 2 (Planned):** Recommendation integration, analytics, gamification

---

## License

This project is developed for academic and research purposes.
