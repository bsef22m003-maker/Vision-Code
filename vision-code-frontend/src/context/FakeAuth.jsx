import { createContext, useContext, useReducer } from "react";
import { useMutation } from "@tanstack/react-query";
import { toast } from "sonner";
import { API_URL } from "../constants";

const AuthContext = createContext();

const initialState = { user: null, isAuthenticated: false, error: null };

function reducer(state, action) {
  switch (action.type) {
    case "login":
      return {
        ...state,
        user: action.payload,
        isAuthenticated: true,
        error: null,
      };
    case "logout":
      return { ...state, user: null, isAuthenticated: false };
    case "signup":
      return { ...state, user: action.payload };
    case "error": // ✅ FIX 1: missing "error" case
      return { ...state, error: action.payload };
    default:
      throw new Error("Unknown action type");
  }
}

function AuthProvider({ children }) {
  const [{ user, isAuthenticated, error }, dispatch] = useReducer(
    reducer,
    initialState,
  );

  async function loginApi({ email, password }) {
    // ✅ FIX 2: destructure object (mutationFn receives one arg)
    const res = await fetch(`${API_URL}/api/auth/token/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);

    const data = await res.json();
    return data;
  }

  async function signupApi({ firstName, lastName, email, password }) {
    // ✅ FIX 2: same here
    const res = await fetch(`${API_URL}/api/auth/signup/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ firstName, lastName, email, password }),
    });

    if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
    return res;
  }

  // ✅ FIX 3: hooks cannot be called inside regular functions — move useMutation to top level
  const loginMutation = useMutation({
    mutationFn: loginApi,
    onSuccess: (data, variables) => {
      localStorage.setItem("access_token", data.access);
      localStorage.setItem("refresh_token", data.refresh);
      dispatch({ type: "login", payload: { email: variables.email } }); // ✅ FIX 4: never store password in state
      toast.success("Login successful 🎉");
    },
    onError: (error) => {
      toast.error(error.message || "Login failed ❌");
      dispatch({ type: "error", payload: "Invalid Credentials" });
    },
  });

  // ✅ FIX 5: similarly hoist signupMutation
  const signupMutation = useMutation({
    mutationFn: signupApi,
    onSuccess: (_, variables) => {
      dispatch({ type: "signup", payload: { email: variables.email } });
      toast.success("Signup successful 🎉");
    },
    onError: (error) => {
      toast.error(error.message || "Signup failed ❌");
    },
  });

  function logout() {
    localStorage.removeItem("access_token"); // ✅ FIX 6: clear tokens on logout
    localStorage.removeItem("refresh_token");
    dispatch({ type: "logout" });
  }

  return (
    <AuthContext.Provider
      value={{
        login: loginMutation, // ✅ FIX 7: expose the mutation object, not a function wrapper
        signup: signupMutation,
        logout,
        user,
        isAuthenticated,
        error,
        dispatch,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined)
    throw new Error("Context was used out of the Auth Provider");
  return context;
}

export { AuthProvider, useAuth };
