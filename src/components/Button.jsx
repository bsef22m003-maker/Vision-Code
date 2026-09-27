function Button({
  children,
  type = "button",
  onClick,
  variant = "primary",
  size = "md",
  fullWidth = false,
  disabled = false,
  className = "",
  ...props
}) {
  const baseStyles =
    "font-semibold rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed";

  const variants = {
    primary:
      "bg-blue-600 text-white hover:bg-blue-700 hover:shadow-lg active:scale-95",
    secondary:
      "bg-white text-blue-600 border border-blue-600 hover:bg-blue-50 active:scale-95",
    outline:
      "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50 hover:shadow-md active:scale-95",
    gradient:
      "bg-gradient-to-r from-blue-600 to-purple-600 text-white hover: shadow-lg active:scale-95",
    danger:
      "bg-red-600 text-white hover:bg-red-700 hover:shadow-lg active:scale-95",
    success:
      "bg-green-600 text-white hover:bg-green-700 hover:shadow-lg active:scale-95",
    ghost: "bg-transparent text-blue-600 hover:bg-blue-50 active:scale-95",
  };

  const sizes = {
    sm: "px-4 py-2 text-sm",
    md: "px-6 py-3 text-base",
    lg: "px-8 py-4 text-lg",
  };

  const widthClass = fullWidth ? "w-full" : "";

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`${baseStyles} ${variants[variant]} ${sizes[size]} ${widthClass} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}

export default Button;
