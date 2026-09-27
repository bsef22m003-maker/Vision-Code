import { useNavigate } from "react-router-dom";
import Navbar from "../components/NavBar";
import { useAuth } from "../context/FakeAuth";

function AppLayout() {
  const { logout } = useAuth();
  const navigate = useNavigate();
  const handleLogout = () => {
    logout();
    navigate("/");
  };
  return (
    <div>
      <Navbar>
        <Button variant="ghost" size="sm" onClick={handleLogout}>
          Logout
        </Button>
      </Navbar>
    </div>
  );
}

export default AppLayout;
