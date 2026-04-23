import { jwtDecode } from "jwt-decode";

export const decodeToken = (token) => {
  try {
    return jwtDecode(token);
  } catch (error) {
    console.error("Invalid token:", error);
    return null;
  }
};

export const isTokenExpired = (token) => {
  try {
    const decoded = jwtDecode(token);
    const currentTime = Date.now() / 1000;
    return decoded.exp < currentTime;
  } catch (error) {
    return true;
  }
};

// Updated for your token structure
export const getTokenData = (token) => {
  if (!token) return null;

  const decoded = decodeToken(token);
  if (!decoded) return null;

  return {
    userId: decoded.user_id, // Note: your token uses user_id, not sub
    tokenType: decoded.token_type || "access",
    expiresAt: decoded.exp ? new Date(decoded.exp * 1000) : null,
    issuedAt: decoded.iat ? new Date(decoded.iat * 1000) : null,
    jti: decoded.jti, // JWT ID for token tracking
  };
};

// Check if token will expire soon (e.g., within 5 minutes)
export const isTokenExpiringSoon = (token, thresholdMinutes = 5) => {
  try {
    const decoded = jwtDecode(token);
    const currentTime = Date.now() / 1000;
    const threshold = thresholdMinutes * 60;
    return decoded.exp - currentTime < threshold;
  } catch (error) {
    return true;
  }
};

// Get token expiry time in seconds
export const getTokenTimeLeft = (token) => {
  try {
    const decoded = jwtDecode(token);
    const currentTime = Date.now() / 1000;
    return Math.max(0, decoded.exp - currentTime);
  } catch (error) {
    return 0;
  }
};
