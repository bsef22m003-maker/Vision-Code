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

export const APP_NAME = "Vision-Code";

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AuthProvider>
          <Routes>
            <Route index element={<HomePage />} />
            <Route path="courses" element={<CoursesPage />} />
            <Route path="course/:id" element={<CourseDetail />} />
            <Route path="community" element={<CommunityPage />} />
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