import { API_URL } from "../constants";

export async function enrollCourse(courseId, token) {
  // Get token from localStorage if not provided
  const authToken = token || localStorage.getItem("authToken");

  if (!authToken) {
    throw new Error("Authentication token not found. Please login first.");
  }

  try {
    const url = `${API_URL}/api/courses/courses/${courseId}/enroll/`;
    console.log("Enrolling in course:", { courseId, url });

    const response = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${authToken}`,
        "Content-Type": "application/json",
      },
    });

    console.log("Enrollment response status:", response.status);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.detail ||
        errorData.message ||
        `HTTP error! status: ${response.status}`
      );
    }

    const data = await response.json();
    console.log("Enrollment successful:", data);
    return data;
  } catch (error) {
    if (error instanceof Error) {
      console.error("Error enrolling in course:", error.message);
      throw new Error(`Enrollment failed: ${error.message}`);
    } else {
      console.error("Unknown error occurred:", error);
      throw new Error("Enrollment failed: Unknown error");
    }
  }
}