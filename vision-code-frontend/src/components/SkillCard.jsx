function SkillCard({
  gradientFrom,
  gradientTo,
  glowFrom,
  glowTo,
  icon,
  title,
  skills,
  delay = "delay-100",
}) {
  return (
    <div className={`relative group scroll-scale ${delay}`}>
      <div
        className={`absolute -inset-0.5 bg-gradient-to-r ${glowFrom} ${glowTo} rounded-xl opacity-0 group-hover:opacity-75 blur-lg transition-all duration-500`}
      ></div>
      <div
        className={`relative bg-gradient-to-br ${gradientFrom} ${gradientTo} p-6 rounded-xl text-white h-full`}
      >
        <div className="w-10 h-10 mb-4 opacity-90">{icon}</div>
        <h3 className="text-xl font-bold mb-2">{title}</h3>
        <ul className="space-y-2 text-sm opacity-90">
          {skills.map((skill, index) => (
            <li key={index}>• {skill}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default SkillCard;
