import Button from "./Button";

function GoogleButton({
  onClick,
  text = "Continue with Google",
  className = "",
}) {
  return (
    <div className="relative group">
      <div className="absolute -inset-0.5 bg-gradient-to-r from-pink-600 via-purple-600 to-blue-600 rounded-lg opacity-0 group-hover:opacity-100 blur-lg transition-all duration-500"></div>
      <div className="relative">
        <Button
          type="button"
          onClick={onClick}
          variant="outline"
          fullWidth
          className={className}
        >
          <span className="flex items-center justify-center gap-3">
            <img
              src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg"
              alt="Google"
              className="w-5 h-5"
            />
            {text}
          </span>
        </Button>
      </div>
    </div>
  );
}

export default GoogleButton;
