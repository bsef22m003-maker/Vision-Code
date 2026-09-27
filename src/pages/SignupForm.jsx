import { useEffect, useState } from "react";
import { useAuth } from "../context/FakeAuth";
import GoogleButton from "../components/GoogleButton";
import Button from "../components/Button";
import { NavLink, useNavigate } from "react-router-dom";
import { toast } from "sonner";

function SignupForm() {
  const [email, setEmail] = useState("info@vision-code.dev");
  const [password, setPassword] = useState("vision-code");
  const [firstName, setFirstName] = useState("First Name");
  const [lastName, setLastName] = useState("Last Name");
  const [role, setRole] = useState("student"); // 👈 NEW
  const [showPassword, setShowPassword] = useState(false);

  const { signup, isAuthenticated } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const data = await signup.mutateAsync({
        email,
        password,
        firstName,
        lastName,
        role,
      });
      navigate("/app");
    } catch (err) {
      console.log("Signup error:", err);
      toast.error("Signup failed. Please try again."); // ✅ NEW: show error toast
    }
  };
  useEffect(() => {
    if (isAuthenticated) {
      navigate("/app", { replace: true });
    }
  }, [isAuthenticated, navigate]);

  return (
    <div className="max-w-md mx-auto mt-20 bg-white p-8 rounded-xl shadow-lg border border-gray-100">
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* First Name */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            First Name
          </label>
          <input
            type="text"
            className="w-full p-3 border border-gray-300 rounded-lg"
            value={firstName}
            onChange={(e) => setFirstName(e.target.value)}
            required
          />
        </div>

        {/* Last Name */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Last Name
          </label>
          <input
            type="text"
            className="w-full p-3 border border-gray-300 rounded-lg"
            value={lastName}
            onChange={(e) => setLastName(e.target.value)}
            required
          />
        </div>

        {/* Email */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Email Address
          </label>
          <input
            type="email"
            className="w-full p-3 border border-gray-300 rounded-lg"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>

        {/* Role Selection 👇 */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Select Role
          </label>

          <div className="flex gap-6">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="radio"
                value="student"
                checked={role === "student"}
                onChange={(e) => setRole(e.target.value)}
              />
              Student
            </label>

            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="radio"
                value="instructor"
                checked={role === "instructor"}
                onChange={(e) => setRole(e.target.value)}
              />
              Instructor
            </label>
          </div>
        </div>

        {/* Password */}
        <div className="relative">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Password
          </label>

          <input
            type={showPassword ? "text" : "password"}
            className="w-full p-3 border border-gray-300 rounded-lg pr-12"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />

          <button
            type="button"
            onClick={() => setShowPassword((prev) => !prev)}
            className="absolute right-3 top-10 text-gray-500"
          >
            {showPassword ? "🙈" : "👁️"}
          </button>
        </div>

        <Button
          type="submit"
          disabled={signup.isPending}
          className="w-full bg-blue-600 text-white py-3 rounded-lg"
        >
          Sign Up
        </Button>
      </form>

      {/* <div className="relative my-6">
        <div className="absolute inset-0 flex items-center">
          <div className="w-full border-t border-gray-300"></div>
        </div>
        <div className="relative flex justify-center text-sm">
          <span className="px-2 bg-white text-gray-500">Or continue with</span>
        </div>
      </div>

      <GoogleButton /> */}

      <p className="mt-4 text-center text-sm text-gray-600">
        Already have an account?{" "}
        <NavLink
          to="/login"
          className="text-blue-600 font-semibold hover:underline"
        >
          Sign in
        </NavLink>
      </p>
    </div>
  );
}

export default SignupForm;
