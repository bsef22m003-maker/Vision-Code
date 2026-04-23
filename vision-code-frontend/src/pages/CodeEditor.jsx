import { useState } from "react";
import { useNavigate } from "react-router-dom";
import Button from "../components/Button";

function CodeEditor() {
  const navigate = useNavigate();
  const [code, setCode] = useState("// Write your code here...\n");
  const [output, setOutput] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [language, setLanguage] = useState("javascript");

  const handleRun = () => {
    setIsRunning(true);
    // Simulate code execution
    setTimeout(() => {
      setOutput("Program executed successfully!\nHello World!");
      setIsRunning(false);
    }, 1000);
  };

  const handleReset = () => {
    setCode("// Write your code here.. .\n");
    setOutput("");
  };

  return (
    <div className="h-screen flex flex-col bg-gray-900">
      {/* Header */}
      <div className="bg-gray-800 border-b border-gray-700 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate("/practice")}
            className="p-2 hover:bg-gray-700 rounded-lg transition"
          >
            <svg
              className="w-5 h-5 text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M10 19l-7-7m0 0l7-7m-7 7h18"
              />
            </svg>
          </button>
          <h1 className="text-white font-semibold">Code Editor</h1>

          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="bg-gray-700 text-white px-3 py-2 rounded-lg text-sm"
          >
            <option value="javascript">JavaScript</option>
            <option value="python">Python</option>
            <option value="java">Java</option>
            <option value="cpp">C++</option>
          </select>
        </div>

        <div className="flex items-center gap-3">
          <Button
            onClick={handleReset}
            variant="outline"
            className="px-4 py-2 text-sm border-gray-600 text-gray-300 hover:bg-gray-700"
          >
            Reset
          </Button>
          <Button
            onClick={handleRun}
            disabled={isRunning}
            className="px-6 py-2 bg-green-600 hover:bg-green-700 text-white text-sm"
          >
            {isRunning ? "Running..." : "Run Code"}
          </Button>
        </div>
      </div>

      {/* Editor & Output */}
      <div className="flex-1 flex overflow-hidden">
        {/* Code Editor */}
        <div className="flex-1 flex flex-col">
          <div className="bg-gray-800 px-4 py-2 text-gray-400 text-sm border-b border-gray-700">
            Code
          </div>
          <textarea
            value={code}
            onChange={(e) => setCode(e.target.value)}
            className="flex-1 bg-gray-900 text-white p-4 font-mono text-sm resize-none focus:outline-none"
            spellCheck="false"
          />
        </div>

        {/* Output Panel */}
        <div className="w-1/3 flex flex-col border-l border-gray-700">
          <div className="bg-gray-800 px-4 py-2 text-gray-400 text-sm border-b border-gray-700">
            Output
          </div>
          <div className="flex-1 bg-gray-900 text-green-400 p-4 font-mono text-sm overflow-auto">
            {output || 'Click "Run Code" to see output... '}
          </div>
        </div>
      </div>
    </div>
  );
}

export default CodeEditor;
