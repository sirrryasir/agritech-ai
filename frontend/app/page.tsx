"use client";
import { useState } from "react";

export default function Home() {
  const [formData, setFormData] = useState({
    N: 90,
    P: 42,
    K: 43,
    temperature: 20.8,
    humidity: 82.0,
    ph: 6.5,
    rainfall: 202.9,
  });

  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

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
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.message || "Failed to predict crop.");
      }

      setPrediction(data.prediction.toUpperCase());
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-white flex flex-col items-center justify-center p-4">
      <div className="max-w-xl w-full bg-slate-800 rounded-2xl shadow-xl overflow-hidden border border-slate-700">
        <div className="p-8">
          <h1 className="text-3xl font-bold bg-linear-to-r from-green-400 to-emerald-500 bg-clip-text text-transparent mb-2">
            AgriTech AI System
          </h1>
          <p className="text-slate-400 mb-8">
            Enter the soil and weather parameters to get the best crop recommendation for your farm.
          </p>

          <form onSubmit={predictCrop} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Nitrogen (N)</label>
                <input required type="number" step="any" name="N" value={formData.N} onChange={handleInputChange} className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-500" />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Phosphorus (P)</label>
                <input required type="number" step="any" name="P" value={formData.P} onChange={handleInputChange} className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-500" />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Potassium (K)</label>
                <input required type="number" step="any" name="K" value={formData.K} onChange={handleInputChange} className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-500" />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Temperature (°C)</label>
                <input required type="number" step="any" name="temperature" value={formData.temperature} onChange={handleInputChange} className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-500" />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Humidity (%)</label>
                <input required type="number" step="any" name="humidity" value={formData.humidity} onChange={handleInputChange} className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-500" />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Soil pH</label>
                <input required type="number" step="any" name="ph" value={formData.ph} onChange={handleInputChange} className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-500" />
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1">Rainfall (mm)</label>
              <input required type="number" step="any" name="rainfall" value={formData.rainfall} onChange={handleInputChange} className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-500" />
            </div>

            <button type="submit" disabled={loading} className="w-full mt-6 bg-linear-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white font-bold py-3 px-4 rounded-lg transition-all transform hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed">
              {loading ? "Analyzing Models..." : "Predict Crop"}
            </button>
          </form>

          {error && (
            <div className="mt-6 p-4 bg-red-900/50 border border-red-500 rounded-lg text-red-200 text-center">
              {error}
            </div>
          )}

          {prediction && (
            <div className="mt-8 text-center animate-in fade-in zoom-in duration-300">
              <p className="text-slate-400 text-sm uppercase tracking-wider mb-2">Recommended Crop</p>
              <div className="inline-block px-8 py-4 bg-linear-to-br from-slate-700 to-slate-800 rounded-2xl shadow-inner border border-slate-600">
                <span className="text-4xl font-black bg-linear-to-r from-green-300 via-emerald-400 to-teal-300 bg-clip-text text-transparent">
                  {prediction}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
