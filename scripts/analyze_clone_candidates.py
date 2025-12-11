"""
Analyze B2B clone candidates for fast ROI and pick the top recommendation.

Clean-room script: no Alpha Loop imports or code reuse.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class Candidate:
    name: str
    category: str
    build_speed: int        # 1-10 (weight 30%)
    revenue_potential: int  # 1-10 (weight 25%)
    marketing: int          # 1-10 (weight 20%)
    market_size: int        # 1-10 (weight 15%)
    defensibility: int      # 1-10 (weight 10%)
    notes: str

    def score(self) -> float:
        return (
            self.build_speed * 0.30
            + self.revenue_potential * 0.25
            + self.marketing * 0.20
            + self.market_size * 0.15
            + self.defensibility * 0.10
        )


def rank_candidates(candidates: List[Candidate]) -> List[Candidate]:
    return sorted(candidates, key=lambda c: c.score(), reverse=True)


def main():
    candidates = [
        Candidate(
            name="InstantMeet (Scheduling + Payments + Team Routing)",
            category="Workflow / Scheduling",
            build_speed=9,
            revenue_potential=8,
            marketing=8,
            market_size=7,
            defensibility=6,
            notes="Fast build; immediate monetization via Stripe; strong PLG via shareable links.",
        ),
        Candidate(
            name="MiniCRM (Lead Inbox + Sequences)",
            category="SaaS Productivity",
            build_speed=7,
            revenue_potential=8,
            marketing=7,
            market_size=8,
            defensibility=6,
            notes="Higher scope; email deliverability risk; still good ROI but slower to harden.",
        ),
        Candidate(
            name="SupportWidget AI (Docs-trained chat + ticket handoff)",
            category="Customer Support",
            build_speed=7,
            revenue_potential=8,
            marketing=7,
            market_size=8,
            defensibility=6,
            notes="Requires vector store + ingestion; more infra; higher perceived value.",
        ),
        Candidate(
            name="Form2Lead (Typeform-lite + Enrichment + Webhooks)",
            category="Workflow Automation",
            build_speed=8,
            revenue_potential=7,
            marketing=8,
            market_size=7,
            defensibility=5,
            notes="Competitive space; fast to build; good SEO/PLG via templates.",
        ),
        Candidate(
            name="DevStatus (Lightweight Incident/Status Page)",
            category="Developer Tools",
            build_speed=8,
            revenue_potential=6,
            marketing=6,
            market_size=6,
            defensibility=5,
            notes="Simple build; niche; lower ticket size unless bundled with uptime.",
        ),
    ]

    ranked = rank_candidates(candidates)
    print("=== Ranked Candidates (Higher is better) ===")
    for c in ranked:
        print(f"{c.name} | Score: {c.score():.2f} | Notes: {c.notes}")

    winner = ranked[0]
    print("\n=== Selected Winner ===")
    print(f"{winner.name} (Category: {winner.category})")
    print(f"Rationale: Fastest to build tonight; immediate monetization via Stripe; strong PLG via shareable booking links; clear B2B value.")


if __name__ == "__main__":
    main()

