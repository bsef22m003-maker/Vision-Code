import { useEffect, useState } from "react";
import { useAuth } from "../context/FakeAuth";
import GoogleButton from "../components/GoogleButton";
import Button from "../components/Button";
import { NavLink, useNavigate } from "react-router-dom";

function LoginForm() {
  const [email, setEmail] = useState("info@vision-code.dev");
  const [password, setPassword] = useState("vision-code");
  const [showPassword, setShowPassword] = useState(false); // 👈 NEW

  const { login, isAuthenticated, error } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const data = await login.mutateAsync({ email, password });
      navigate("/dashboard");
    } catch (err) {
      console.log("Login error:", err);
    }
  };

  useEffect(() => {
    if (isAuthenticated) {
      navigate("/app", { replace: true });
    }
  }, [isAuthenticated, navigate]);

  return (
    <div className="max-w-md mx-auto mt-20 bg-white p-8 rounded-xl shadow-lg border border-gray-100">
      <h2 className="text-3xl font-bold mb-6 text-center text-gray-900">
        Welcome Back
      </h2>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Email Address
          </label>
          <input
            type="email"
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>

        {/* 👇 PASSWORD FIELD WITH TOGGLE */}
        <div className="relative">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Password
          </label>

          <input
            type={showPassword ? "text" : "password"} // 👈 toggle type
            className="w-full p-3 border border-gray-300 rounded-lg pr-12 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />

          {/* 👇 Toggle Button */}
          <button
            type="button"
            onClick={() => setShowPassword((prev) => !prev)}
            className="absolute right-3 top-10 text-gray-500 hover:text-gray-700"
          >
            {showPassword ? "🙈" : "👁️"}
          </button>
        </div>

        <Button
          type="submit"
          disabled={login.isPending}
          className="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 font-semibold transition"
        >
          {login.isPending ? "Signing in..." : "Sign In"}
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
        Don't have an account?{" "}
        <NavLink
          to="/signup"
          className="text-blue-600 font-semibold hover:underline"
        >
          Sign up
        </NavLink>
      </p>
    </div>
  );
}

export default LoginForm;
