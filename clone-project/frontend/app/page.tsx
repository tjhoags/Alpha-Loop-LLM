"use client";

import { useState } from "react";

export default function Home() {
  const [email, setEmail] = useState("");
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitted(true);
  };

  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-50 to-white">
      <div className="max-w-5xl mx-auto px-6 py-16">
        <header className="flex items-center justify-between mb-12">
          <div className="text-2xl font-bold text-slate-900">InstantMeet</div>
          <div className="flex gap-4 text-sm">
            <a className="text-slate-700 hover:text-slate-900" href="#features">Features</a>
            <a className="text-slate-700 hover:text-slate-900" href="#pricing">Pricing</a>
            <a className="text-slate-700 hover:text-slate-900" href="/dashboard">Dashboard</a>
          </div>
        </header>

        <section className="grid md:grid-cols-2 gap-10 items-center mb-16">
          <div>
            <p className="text-sm font-semibold text-blue-600 mb-3">Payments-first scheduling</p>
            <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-6">
              Get paid before you meet. Route every lead to the right teammate.
            </h1>
            <p className="text-lg text-slate-700 mb-6">
              InstantMeet combines booking links, payments, and team routing so you never waste time on low-intent calls.
            </p>
            <form onSubmit={handleSubmit} className="flex flex-col sm:flex-row gap-3">
              <input
                type="email"
                required
                placeholder="you@company.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="flex-1 rounded-lg border border-slate-200 px-4 py-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button
                type="submit"
                className="rounded-lg bg-blue-600 px-5 py-3 text-white font-semibold shadow-lg hover:bg-blue-700"
              >
                Start free
              </button>
            </form>
            {submitted && (
              <p className="text-sm text-green-700 mt-3">Thanks! We’ll email your access link.</p>
            )}
            <div className="flex gap-4 mt-6 text-sm text-slate-600">
              <span>Payments-first</span>
              <span>Team routing</span>
              <span>Embeddable widget</span>
            </div>
          </div>
          <div className="bg-white shadow-xl border border-slate-100 rounded-2xl p-6">
            <p className="text-sm font-semibold text-slate-600 mb-2">Booking preview</p>
            <div className="rounded-xl border border-slate-200 p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-semibold text-slate-900">Paid Discovery Call</p>
                  <p className="text-sm text-slate-600">30 minutes · $75</p>
                </div>
                <span className="px-3 py-1 text-xs rounded-full bg-blue-50 text-blue-700">Instant pay</span>
              </div>
              <div className="space-y-2">
                <p className="text-sm text-slate-700">Pick a time</p>
                <div className="grid grid-cols-3 gap-2">
                  {["Mon 10:00", "Mon 14:00", "Tue 11:30"].map((slot) => (
                    <button key={slot} className="rounded-lg border border-slate-200 py-2 text-sm hover:border-blue-500">
                      {slot}
                    </button>
                  ))}
                </div>
              </div>
              <button className="w-full rounded-lg bg-slate-900 text-white py-3 font-semibold hover:bg-slate-800">
                Continue to payment
              </button>
            </div>
          </div>
        </section>

        <section id="features" className="grid md:grid-cols-3 gap-6 mb-16">
          {[
            {
              title: "Payments-first",
              desc: "Collect payment before confirming. Reduce no-shows and filter for intent.",
            },
            {
              title: "Team routing",
              desc: "Round-robin or priority routing so every lead goes to the right teammate.",
            },
            {
              title: "Embed anywhere",
              desc: "Drop your booking widget into landing pages, proposals, and portals.",
            },
          ].map((f) => (
            <div key={f.title} className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
              <h3 className="font-semibold text-slate-900 mb-2">{f.title}</h3>
              <p className="text-sm text-slate-700">{f.desc}</p>
            </div>
          ))}
        </section>

        <section id="pricing" className="bg-white border border-slate-200 rounded-2xl p-8 shadow-sm">
          <h2 className="text-2xl font-bold text-slate-900 mb-4">Pricing</h2>
          <div className="grid md:grid-cols-3 gap-4">
            {[
              { name: "Starter", price: "$19/mo", features: ["1 calendar", "Basic branding", "Email notifications"] },
              { name: "Pro", price: "$49/mo", features: ["Team routing", "Advanced branding", "Webhooks + CRM export"] },
              { name: "Team", price: "$99/mo", features: ["Workspaces", "Audit logs", "Priority support"] },
            ].map((tier) => (
              <div key={tier.name} className="rounded-xl border border-slate-200 p-5 shadow-sm">
                <p className="text-sm font-semibold text-blue-600 mb-1">{tier.name}</p>
                <p className="text-2xl font-bold text-slate-900 mb-3">{tier.price}</p>
                <ul className="space-y-1 text-sm text-slate-700">
                  {tier.features.map((feat) => (
                    <li key={feat}>• {feat}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}

