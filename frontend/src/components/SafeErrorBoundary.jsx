// components/SafeErrorBoundary.jsx
import { Component } from "react";

class SafeErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      fallbackPath: props.fallbackPath || "/",
    };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({ errorInfo });
    console.error("Error caught by boundary:", error, errorInfo);

    // Сохраняем предыдущий маршрут в sessionStorage
    const previousPath = window.location.hash;
    sessionStorage.setItem("lastSafePath", previousPath);
  }

  handleGoBack = () => {
    const safePath =
      sessionStorage.getItem("lastSafePath") || this.state.fallbackPath;
    window.location.hash = safePath;
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  handleReset = () => {
    window.location.hash = this.state.fallbackPath;
    window.location.reload(); // полный сброс
  };

  render() {
    if (this.state.hasError) {
      return (
        <div
          className="card error-boundary"
          style={{ textAlign: "center", padding: "2rem" }}
        >
          <h2>⚠️ Что-то пошло не так</h2>
          <details style={{ margin: "1rem 0", textAlign: "left" }}>
            <summary>Технические детали</summary>
            <pre style={{ fontSize: "12px", overflow: "auto" }}>
              {this.state.error?.toString()}
              {"\n\n"}
              {this.state.errorInfo?.componentStack}
            </pre>
          </details>
          <div
            style={{ display: "flex", gap: "1rem", justifyContent: "center" }}
          >
            <button onClick={this.handleGoBack} className="primary">
              ↩️ Вернуться назад
            </button>
            <button onClick={this.handleReset}>
              🔄 Перезагрузить приложение
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default SafeErrorBoundary;
