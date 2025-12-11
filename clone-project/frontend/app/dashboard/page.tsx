"use client";

import { useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8001";

export default function Dashboard() {
  const [meetingName, setMeetingName] = useState("Paid Discovery Call");
  const [price, setPrice] = useState(75);
  const [duration, setDuration] = useState(30);
  const [status, setStatus] = useState("");

  const handleCreateMeetingType = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus("Saving...");
    try {
      const resp = await fetch(`${API_BASE}/meeting-types`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({
          user_id: "1",
          name: meetingName,
          price_cents: String(price * 100),
          duration_minutes: String(duration),
          description: "Auto-created from dashboard",
          team_routing: "solo",
        }),
      });
      if (!resp.ok) throw new Error("Failed to save");
      setStatus("Meeting type created. Copy your booking link: /book/1");
    } catch (err) {
      setStatus("Error creating meeting type.");
    }
  };

  return (
    <main className="min-h-screen bg-slate-50">
      <div className="max-w-4xl mx-auto px-6 py-10">
        <header className="flex items-center justify-between mb-8">
          <div>
            <p className="text-sm text-slate-600">Dashboard</p>
            <h1 className="text-3xl font-bold text-slate-900">InstantMeet</h1>
          </div>
          <a href="/" className="text-sm text-blue-600">Back to landing</a>
        </header>

        <section className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm mb-6">
          <h2 className="text-xl font-semibold text-slate-900 mb-2">Create a meeting type</h2>
          <p className="text-sm text-slate-600 mb-4">Collect payments and route leads automatically.</p>
          <form className="space-y-4" onSubmit={handleCreateMeetingType}>
            <div>
              <label className="block text-sm text-slate-700 mb-1">Name</label>
              <input
                value={meetingName}
                onChange={(e) => setMeetingName(e.target.value)}
                className="w-full rounded-lg border border-slate-200 px-3 py-2"
                required
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-slate-700 mb-1">Price (USD)</label>
                <input
                  type="number"
                  value={price}
                  onChange={(e) => setPrice(Number(e.target.value))}
                  className="w-full rounded-lg border border-slate-200 px-3 py-2"
                  min={0}
                />
              </div>
              <div>
                <label className="block text-sm text-slate-700 mb-1">Duration (minutes)</label>
                <input
                  type="number"
                  value={duration}
                  onChange={(e) => setDuration(Number(e.target.value))}
                  className="w-full rounded-lg border border-slate-200 px-3 py-2"
                  min={15}
                />
              </div>
            </div>
            <button
              type="submit"
              className="rounded-lg bg-blue-600 text-white px-4 py-2 font-semibold hover:bg-blue-700"
            >
              Save meeting type
            </button>
          </form>
          {status && <p className="text-sm text-slate-700 mt-3">{status}</p>}
        </section>

        <section className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
          <h2 className="text-xl font-semibold text-slate-900 mb-2">Bookings</h2>
          <p className="text-sm text-slate-600 mb-4">Connect your booking link to start seeing paid meetings.</p>
          <div className="p-4 rounded-lg border border-dashed border-slate-300 text-sm text-slate-700">
            No bookings yet. Share your link: <code className="text-blue-700">/book/1</code>
          </div>
        </section>
      </div>
    </main>
  );
}

