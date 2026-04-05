const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8001";
async function request(url, options = {}) {
  const res = await fetch(`${API_BASE}${url}`, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}
export async function healthcheck() {
  return request("/health");
}
export async function uploadForTrain(file) {
    const form = new FormData();
    form.append("file", file);
    return request("/api/ml/train", { method: "POST", body: form });
  }
export async function uploadForShap(file, modelId = "XGB") {
    const form = new FormData();
    form.append("file", file);
    return request(`/api/ml/shap?model_id=${encodeURIComponent(modelId)}`, {
      method: "POST",
      body: form
    });
}
export async function getTaskStatus(taskId) {
    return request(`/api/ml/tasks/${taskId}`);
}

export async function uploadForCorrelation(file) {
    const form = new FormData();
    form.append("file", file);
    return request("/api/ml/correlation", { method: "POST", body: form });
}
