import { Navigate } from "react-router-dom";
import { jwtDecode } from "jwt-decode";
import InstructorDashboard from "./InstructorDashboard";
function ProtectedRouter({ children }) {
  const token = localStorage.getItem("access_token");
  if (!token) return <Navigate to="/" replace />;
  const decodedToken = jwtDecode(token);

  const role = decodedToken?.role;
  return role === "instructor" ? <InstructorDashboard /> : children;
}

export default ProtectedRouter;
