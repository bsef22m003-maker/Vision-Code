import { BrowserRouter, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import HomePage from "./pages/HomePage";
import { AuthProvider } from "./context/FakeAuth";
import LoginForm from "./pages/LoginForm";
import ProtectedRouter from "./pages/ProtectedRouter";
import Dashboard from "./pages/Dashboard";
import CoursesPage from "./pages/CoursesPage";
import CommunityPage from "./pages/CommunityPage";
import CourseDetail from "./pages/CourseDetail";
import { Toaster } from "sonner";
import StudentCourseDetail from "./pages/StudentCourseDetail";
import QuizPage from "./pages/QuizPage";
import InstructorDashboard from "./pages/InstructorDashboard";
import CreateCoursePage from "./pages/CreateCourse";
import ProctoringComponent from "./pages/ProctoringComponent";
import InstructorCourseDetail from "./pages/InstructorCourseDetail";
export const APP_NAME = "Vision-Code";

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AuthProvider>
          <Toaster position="top-right" reverseOrder={false} />
          <Routes>
            <Route index element={<HomePage />} />
            <Route path="courses" element={<CoursesPage />} />
            <Route path="course/:id" element={<CourseDetail />} />
            <Route path="community" element={<CommunityPage />} />
            <Route
              path="/student/course/:courseId"
              element={<StudentCourseDetail />}
            />
            <Route path="/proctoring" element={<ProctoringComponent />} />
            <Route
              path="/instructor/create-course"
              element={<CreateCoursePage />}
            />
            <Route
              path="/instructor/course/:courseId"
              element={<InstructorCourseDetail />}
            />
            <Route path="/instructor" element={<InstructorDashboard />} />
            <Route path="/student/quiz/:id" element={<QuizPage />} />
            <Route
              path="app"
              element={
                <ProtectedRouter>
                  <Dashboard />
                </ProtectedRouter>
              }
            />
            <Route path="login" element={<LoginForm />} />
          </Routes>
        </AuthProvider>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
