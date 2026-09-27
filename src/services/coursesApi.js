import { API_URL } from "../constants";
export async function callCoursesApi() {
  try {
    const response = await fetch(`${API_URL}/api/courses/courses/`);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log("Courses data:", data);
    return data;
  } catch (error) {
    if (error instanceof Error) {
      console.error("Error fetching courses:", error.message);
    } else {
      console.error("Unknown error occurred:", error);
    }
    throw error;
  }
}

