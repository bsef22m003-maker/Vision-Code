import { Navigate } from "react-router-dom";
import { jwtDecode } from "jwt-decode";
import InstructorDashboard from "./InstructorDashboard";
function ProtectedRouter({ children }) {
  const token = localStorage.getItem("access_token");
  const decodedToken = jwtDecode(token);
  console.log("Decoded token:", decodedToken);
  const role = decodedToken?.role;
  console.log("Decoded token:", role);
  if (!token) return <Navigate to="/" replace />;
  return role === "instructor" ? <InstructorDashboard /> : children;
}

export default ProtectedRouter;
