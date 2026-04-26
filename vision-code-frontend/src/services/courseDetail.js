import { API_URL } from "../constants";

export async function getCourseDetail(courseId) {
  try {
    const response = await fetch(
      `${API_URL}/api/courses/courses/${courseId}/`
    );

    if (!response.ok) {
      throw new Error(
        `HTTP error! status: ${response.status} - Course not found`
      );
    }

    const data = await response.json();
    console.log("Course Detail:", data);
    return data;
  } catch (error) {
    if (error instanceof Error) {
      console.error("Error fetching course detail:", error.message);
    } else {
      console.error("Unknown error occurred:", error);
    }
    throw error;
  }
}