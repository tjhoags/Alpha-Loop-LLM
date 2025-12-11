"use client";

import { useState } from "react";
import { useParams } from "next/navigation";

const API_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8001";

export default function BookPage() {
  const params = useParams();
  const meetingTypeId = params?.id as string;
  const [slot, setSlot] = useState("2025-12-11T15:00");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [status, setStatus] = useState("");

  const handleBook = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus("Booking...");
    try {
      const resp = await fetch(`${API_BASE}/book`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({
          meeting_type_id: meetingTypeId,
          attendee_name: name,
          attendee_email: email,
          start_time: slot,
        }),
      });
      if (!resp.ok) throw new Error("Booking failed");
      setStatus("Booked! Check your email for confirmation.");
    } catch {
      setStatus("Error booking. Try again.");
    }
  };

  return (
    <main className="min-h-screen bg-slate-50">
      <div className="max-w-xl mx-auto px-6 py-12">
        <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
          <p className="text-sm text-slate-600 mb-2">Book a time</p>
          <h1 className="text-2xl font-bold text-slate-900 mb-4">Paid Discovery Call</h1>
          <p className="text-sm text-slate-700 mb-4">30 minutes · $75</p>
          <form className="space-y-4" onSubmit={handleBook}>
            <div>
              <label className="block text-sm text-slate-700 mb-1">Select a slot</label>
              <select
                value={slot}
                onChange={(e) => setSlot(e.target.value)}
                className="w-full rounded-lg border border-slate-200 px-3 py-2"
              >
                <option value="2025-12-11T15:00">Thu 3:00 PM</option>
                <option value="2025-12-11T16:00">Thu 4:00 PM</option>
                <option value="2025-12-12T14:00">Fri 2:00 PM</option>
              </select>
            </div>
            <div>
              <label className="block text-sm text-slate-700 mb-1">Name</label>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full rounded-lg border border-slate-200 px-3 py-2"
                required
              />
            </div>
            <div>
              <label className="block text-sm text-slate-700 mb-1">Email</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full rounded-lg border border-slate-200 px-3 py-2"
                required
              />
            </div>
            <button
              type="submit"
              className="w-full rounded-lg bg-blue-600 text-white px-4 py-2 font-semibold hover:bg-blue-700"
            >
              Confirm & pay
            </button>
          </form>
          {status && <p className="text-sm text-slate-700 mt-3">{status}</p>}
        </div>
      </div>
    </main>
  );
}

