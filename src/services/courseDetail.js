import { API_URL } from "../constants";

export async function getCourseDetail(courseId) {
  const startTime = new Date().getTime();
  try {
    const response = await fetch(`${API_URL}/api/courses/courses/${courseId}/`);

    if (!response.ok) {
      throw new Error(
        `HTTP error! status: ${response.status} - Course not found`,
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
  } finally {
    const endTime = new Date().getTime();
    const executionTime = endTime - startTime;
    console.log(`getCourseDetail execution time: ${executionTime / 1000} s`);
  }
}
export async function getCourseModules(moduleId) {
  try {
    const response = await fetch(
      `${API_URL}/api/courses/course-modules/${moduleId}/`,
    );
    const data = await response.json();
    console.log("Course Modules:", data);
    return data;
  } catch (error) {
    if (error instanceof Error) {
      console.error("Error fetching course modules:", error.message);
    } else {
      console.error("Unknown error occurred:", error);
    }
  }
}
