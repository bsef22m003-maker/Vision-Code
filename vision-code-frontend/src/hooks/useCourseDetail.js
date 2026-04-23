import { useQuery } from "@tanstack/react-query";
import { getCourseDetail } from "../services/courseDetail";
export function useCourseDetail(courseId) {
  return useQuery({
    queryKey: ["course", courseId],
    queryFn: () => getCourseDetail(courseId),
    enabled: !!courseId,
  });
}