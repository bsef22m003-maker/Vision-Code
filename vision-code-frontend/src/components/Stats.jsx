function decorateNums(...variables) {
  return variables.map((element) => {
    let base = element - (element % 10);
    let str = base > 1000 ? base / 1000 + "k" : base;
    return str + "+";
  });
}
function Stats({
  studentsNum = 50001,
  coursesNum = 123,
  communitySoltions = 10001,
  avgRating = 4.9,
}) {
  const [students, courses, communitySols] = decorateNums(
    studentsNum,
    coursesNum,
    communitySoltions
  );
  return (
    <div className="bg-white border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
          <div className="scroll-scale delay-100">
            <div className="text-3xl font-bold text-gray-900">{students}</div>
            <div className="text-gray-500">Students</div>
          </div>
          <div className="scroll-scale delay-200">
            <div className="text-3xl font-bold text-gray-900">{courses}</div>
            <div className="text-gray-500">Courses</div>
          </div>
          <div className="scroll-scale delay-300">
            <div className="text-3xl font-bold text-gray-900">
              {communitySols}
            </div>
            <div className="text-gray-500">Community Solutions</div>
          </div>
          <div className="scroll-scale delay-400">
            <div className="text-3xl font-bold text-gray-900">
              {avgRating}/5
            </div>
            <div className="text-gray-500">Average Rating</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Stats;
