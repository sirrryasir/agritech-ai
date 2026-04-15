"use client";
import { useState } from "react";

// Feature info descriptions
const featureInfo: Record<string, { label: string; unit: string; description: string; min: number; max: number }> = {
  N: { label: "Nitrogen (N)", unit: "kg/ha", description: "Nitrogen content in soil", min: 0, max: 140 },
  P: { label: "Phosphorus (P)", unit: "kg/ha", description: "Phosphorus content in soil", min: 5, max: 145 },
  K: { label: "Potassium (K)", unit: "kg/ha", description: "Potassium content in soil", min: 5, max: 205 },
  temperature: { label: "Temperature", unit: "C", description: "Average temperature", min: 8, max: 44 },
  humidity: { label: "Humidity", unit: "%", description: "Relative humidity level", min: 14, max: 100 },
  ph: { label: "Soil pH", unit: "pH", description: "pH value of soil", min: 3.5, max: 10 },
  rainfall: { label: "Rainfall", unit: "mm", description: "Annual rainfall amount", min: 20, max: 300 },
};

export default function Home() {
  const [formData, setFormData] = useState({
    N: 90, P: 42, K: 43,
    temperature: 20.8, humidity: 82.0, ph: 6.5, rainfall: 202.9,
  });

  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState<"predict" | "how" | "about">("predict");

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: parseFloat(e.target.value),
    });
  };

  const predictCrop = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setPrediction(null);
    setError(null);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";
      const response = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.message || "Failed to predict crop.");
      }

      setPrediction(data.prediction);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans">
      {/* Header */}
      <header className="bg-slate-900 border-b border-slate-800">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold text-white tracking-tight">AgriTech AI</h1>
            <p className="text-xs text-slate-500 uppercase tracking-widest mt-1">Crop Recommendation Model</p>
          </div>
          <nav className="hidden md:flex gap-2">
            {(["predict", "how", "about"] as const).map((section) => (
              <button
                key={section}
                onClick={() => setActiveSection(section)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  activeSection === section
                    ? "bg-slate-800 text-white border border-slate-700"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                {section === "predict" ? "Predict" : section === "how" ? "Documentation" : "About"}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* Mobile Navigation */}
      <div className="md:hidden flex border-b border-slate-800 bg-slate-900">
        {(["predict", "how", "about"] as const).map((section) => (
          <button
            key={section}
            onClick={() => setActiveSection(section)}
            className={`flex-1 py-4 text-sm font-medium border-b-2 transition-colors ${
              activeSection === section
                ? "text-white border-white bg-slate-800"
                : "text-slate-500 border-transparent"
            }`}
          >
            {section === "predict" ? "Predict" : section === "how" ? "Docs" : "About"}
          </button>
        ))}
      </div>

      <main className="max-w-6xl mx-auto px-6 py-12">
        {/* =================== PREDICT SECTION =================== */}
        {activeSection === "predict" && (
          <div className="space-y-12">
            {/* Hero */}
            <div className="space-y-4 max-w-2xl">
              <h2 className="text-3xl font-bold text-white tracking-tight">
                Data Input Parameters
              </h2>
              <p className="text-slate-400 leading-relaxed">
                Enter your soil nutrients and weather conditions below. The model will analyze the data and predict the optimal crop for cultivation.
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
              {/* Form */}
              <div className="lg:col-span-3">
                <form onSubmit={predictCrop} className="space-y-8">
                  <div className="bg-slate-900 border border-slate-800 rounded-lg p-8">
                    <h3 className="text-sm font-semibold text-white uppercase tracking-widest mb-6">
                      Soil Composition
                    </h3>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
                      {["N", "P", "K"].map((key) => (
                        <div key={key}>
                          <label className="block text-sm font-medium text-slate-300 mb-2">
                            {featureInfo[key].label}
                            <span className="text-slate-500 ml-1">({featureInfo[key].unit})</span>
                          </label>
                          <input
                            required
                            type="number"
                            step="any"
                            name={key}
                            value={formData[key as keyof typeof formData]}
                            onChange={handleInputChange}
                            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-3 text-white placeholder-slate-600 focus:outline-none focus:border-green-500 focus:ring-1 focus:ring-green-500 transition-colors"
                          />
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="bg-slate-900 border border-slate-800 rounded-lg p-8">
                    <h3 className="text-sm font-semibold text-white uppercase tracking-widest mb-6">
                      Environmental Metrics
                    </h3>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                      {["temperature", "humidity", "ph", "rainfall"].map((key) => (
                        <div key={key}>
                          <label className="block text-sm font-medium text-slate-300 mb-2">
                            {featureInfo[key].label}
                            <span className="text-slate-500 ml-1">({featureInfo[key].unit})</span>
                          </label>
                          <input
                            required
                            type="number"
                            step="any"
                            name={key}
                            value={formData[key as keyof typeof formData]}
                            onChange={handleInputChange}
                            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-3 text-white placeholder-slate-600 focus:outline-none focus:border-green-500 focus:ring-1 focus:ring-green-500 transition-colors"
                          />
                        </div>
                      ))}
                    </div>
                  </div>

                  <button
                    type="submit"
                    disabled={loading}
                    className="w-full bg-green-600 hover:bg-green-500 disabled:bg-slate-800 disabled:text-slate-500 text-white font-semibold py-4 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
                  >
                    {loading ? "Processing Inference..." : "Execute Model"}
                  </button>
                </form>
              </div>

              {/* Result Panel */}
              <div className="lg:col-span-2">
                <div className="bg-slate-900 border border-slate-800 rounded-lg p-8 h-full min-h-[300px] flex flex-col justify-center items-center">
                  {!prediction && !error && !loading && (
                    <div className="text-slate-500 text-sm text-center">
                      <p>Enter parameters and execute the model to see the recommendation.</p>
                    </div>
                  )}

                  {loading && (
                    <div className="text-slate-400 text-sm text-center">
                      <p>Running Random Forest inference...</p>
                    </div>
                  )}

                  {error && (
                    <div className="w-full p-4 bg-slate-950 border border-red-900/50 rounded-lg text-red-400 text-sm text-center">
                      <span className="font-semibold block mb-1">Execution Error</span>
                      {error}
                    </div>
                  )}

                  {prediction && (
                    <div className="text-center w-full">
                      <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-4">Inference Output</p>
                      <div className="py-6 border-y border-slate-800 mb-6">
                        <span className="text-4xl font-bold text-white tracking-tight">
                          {prediction.toUpperCase()}
                        </span>
                      </div>
                      <p className="text-xs text-slate-500">
                        Method: Random Forest Ensemble
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* =================== HOW IT WORKS SECTION =================== */}
        {activeSection === "how" && (
          <div className="space-y-12 max-w-3xl">
            <div className="space-y-4">
              <h2 className="text-3xl font-bold text-white tracking-tight">
                Technical Documentation
              </h2>
              <p className="text-slate-400 leading-relaxed">
                Overview of the machine learning pipeline and algorithms used in the AgriTech recommendation system.
              </p>
            </div>

            <div className="space-y-6">
              {[
                { step: "01", title: "Data Collection", desc: "Dataset contains 2,200 observation samples with 7 target features (N, P, K, temperature, humidity, pH, rainfall) and 22 distinct crop classifications." },
                { step: "02", title: "Preprocessing", desc: "Target labels are numerically encoded using LabelEncoder. Input features are standardized via StandardScaler (Z-score normalization) to ensure uniform metric weighting." },
                { step: "03", title: "Train/Validation Split", desc: "Data is split 80/20 with stratification, maintaining class representation ratios across both training and testing datasets." },
                { step: "04", title: "Model Architecture", desc: "Trained using a Random Forest ensemble. Hyperparameters optimized through GridSearchCV (144 parameter combinations executed across 5 cross-validation folds)." },
                { step: "05", title: "Evaluation Metrics", desc: "Model performance assessed via Accuracy Score, Classification Report (Precision/Recall/F-1 Score), and Confusion Matrix verification." },
                { step: "06", title: "API Deployment", desc: "Persisted model artifacts (.joblib) are served via a Flask REST API, interfaced with this Next.js frontend." },
              ].map((item) => (
                <div key={item.step} className="p-6 bg-slate-900 border border-slate-800 rounded-lg flex gap-6">
                  <div className="text-sm font-bold text-slate-500 mt-0.5">
                    {item.step}
                  </div>
                  <div>
                    <h3 className="text-white font-semibold mb-2">{item.title}</h3>
                    <p className="text-slate-400 text-sm leading-relaxed">{item.desc}</p>
                  </div>
                </div>
              ))}
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 pt-6">
              <div className="p-6 bg-slate-900 border border-slate-800 rounded-lg">
                <h3 className="text-white font-semibold mb-2">Gaussian Naive Bayes</h3>
                <p className="text-slate-400 text-sm mb-4">Baseline probabilistic classification model.</p>
                <div className="text-2xl font-bold text-slate-300">99.55%</div>
                <p className="text-xs text-slate-500 mt-1">Test Accuracy</p>
              </div>
              <div className="p-6 bg-slate-900 border border-slate-700/50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-white font-semibold">Random Forest</h3>
                  <span className="text-[10px] bg-slate-800 text-slate-300 px-2 py-1 rounded font-bold uppercase">Production</span>
                </div>
                <p className="text-slate-400 text-sm mb-4">Tree ensemble with hyperparameter tuning.</p>
                <div className="text-2xl font-bold text-white">99.55%</div>
                <p className="text-xs text-slate-500 mt-1">Cross-Validation Mean</p>
              </div>
            </div>
          </div>
        )}

        {/* =================== ABOUT SECTION =================== */}
        {activeSection === "about" && (
          <div className="space-y-12 max-w-3xl">
            <div className="space-y-4">
              <h2 className="text-3xl font-bold text-white tracking-tight">
                Project Overview
              </h2>
              <p className="text-slate-400">Final Capstone Project -- DS & ML Bootcamp 2026</p>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              {[
                { label: "Data Source", value: "Kaggle" },
                { label: "Samples", value: "2,200" },
                { label: "Features", value: "7" },
                { label: "Crop Classes", value: "22" },
              ].map((item) => (
                <div key={item.label} className="p-4 bg-slate-900 border border-slate-800 rounded-lg">
                  <div className="text-xl font-bold text-white mb-1">{item.value}</div>
                  <div className="text-xs font-medium text-slate-500 uppercase">{item.label}</div>
                </div>
              ))}
            </div>

            <div className="p-8 bg-slate-900 border border-slate-800 rounded-lg space-y-8">
              <div>
                <h3 className="text-white font-semibold mb-3">Problem Architecture</h3>
                <p className="text-slate-400 text-sm leading-relaxed">
                  Agriculture currently relies heavily on traditional intuition. This system replaces intuition with data-driven modeling, mapping soil nutrient profiles (N, P, K) and environmental metrics against a dataset of optimal growth conditions.
                </p>
              </div>

              <div>
                <h3 className="text-white font-semibold mb-4">Key Technical Learnings</h3>
                <ul className="space-y-5 text-sm text-slate-400">
                  <li className="flex gap-4">
                    <span className="text-slate-600 font-bold mt-0.5">01</span>
                    <span className="leading-relaxed">State persistence using <code className="text-slate-300">joblib</code> is mandatory for deploying standardization (StandardScaler) and label mapping (LabelEncoder) alongside the classifier model.</span>
                  </li>
                  <li className="flex gap-4">
                    <span className="text-slate-600 font-bold mt-0.5">02</span>
                    <span className="leading-relaxed">Cross-Origin Resource Sharing (CORS) policies require explicit configuration in the Flask REST API layer to permit external client requests.</span>
                  </li>
                  <li className="flex gap-4">
                    <span className="text-slate-600 font-bold mt-0.5">03</span>
                    <span className="leading-relaxed">Classical ML ensembles (Random Forest) demonstrate sufficient classification accuracy for structured tabular data, avoiding the computational overhead of neural networks.</span>
                  </li>
                  <li className="flex gap-4">
                    <span className="text-slate-600 font-bold mt-0.5">04</span>
                    <span className="leading-relaxed">Hyperparameter tuning via GridSearchCV combined with K-Fold cross-validation prevents overfitting and ensures generalization metrics.</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 mt-16">
        <div className="max-w-6xl mx-auto px-6 py-8 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-slate-500 text-xs font-medium">
            AgriTech AI -- By <a href="https://yaasir.dev" target="_blank" rel="noreferrer" className="text-slate-400 hover:text-white text-xs font-medium transition-colors">Yasir Hassan Ahmed</a>
          </p>
          <div className="flex gap-6">
            <a href="https://github.com/sirrryasir/agritech-ai" target="_blank" rel="noreferrer"
              className="text-slate-400 hover:text-white text-xs font-medium transition-colors">
              Source Code
            </a>
            <a href="https://github.com/goobolabs/feb-ds-ml-bootcamp-2026" target="_blank" rel="noreferrer"
              className="text-slate-400 hover:text-white text-xs font-medium transition-colors">
              Bootcamp
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
