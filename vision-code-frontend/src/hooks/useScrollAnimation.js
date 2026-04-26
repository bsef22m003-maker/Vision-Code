import { useEffect } from "react";

function useScrollAnimation() {
  useEffect(() => {
    const animatedElements = document.querySelectorAll(
      ".scroll-fade-up, .scroll-slide-left, .scroll-slide-right, .scroll-scale"
    );

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("revealed");
          } else {
            entry.target.classList.remove("revealed");
          }
        });
      },
      {
        threshold: 0.1,
        rootMargin: "0px 0px -50px 0px",
      }
    );

    animatedElements.forEach((element) => {
      observer.observe(element);
    });

    // Cleanup
    return () => {
      animatedElements.forEach((element) => {
        observer.unobserve(element);
      });
    };
  }, []);
}

export default useScrollAnimation;
